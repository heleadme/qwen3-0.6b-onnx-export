"""Bind the pinned Qwen3-0.6B weights to an existing ONNX graph.

Requires onnx, numpy, and the pinned model.safetensors source file. This
script writes a self-contained ONNX. Run directly for decode, or call
main("prefill") for prefill.
"""

import hashlib
import json
import os
import struct
import tempfile
from pathlib import Path

import numpy as np
import onnx


ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "model" / "model.safetensors"
CONFIG = ROOT / "model" / "config.json"
MANIFEST = ROOT / "outputs" / "manifest.json"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
WEIGHTS_SHA256 = "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b"
CONFIG_SHA256 = "660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd"
ROPE_NAME = "model.rotary_emb.inv_freq"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def value_shape(value):
    dimensions = value.type.tensor_type.shape.dim
    if any(not dimension.HasField("dim_value") for dimension in dimensions):
        raise ValueError(f"Non-fixed weight shape: {value.name}")
    return [dimension.dim_value for dimension in dimensions]


def check_tied_embeddings(tensors, data_start):
    first = tensors["model.embed_tokens.weight"]["data_offsets"]
    second = tensors["lm_head.weight"]["data_offsets"]
    if first[1] - first[0] != second[1] - second[0]:
        raise ValueError("Tied embedding lengths differ")
    remaining = first[1] - first[0]
    with WEIGHTS.open("rb") as left, WEIGHTS.open("rb") as right:
        left.seek(data_start + first[0])
        right.seek(data_start + second[0])
        while remaining:
            count = min(8 * 1024 * 1024, remaining)
            if left.read(count) != right.read(count):
                raise ValueError("Tied embeddings differ")
            remaining -= count


def main(component="decode"):
    if component not in ("decode", "prefill"):
        raise ValueError(f"Unknown ONNX component: {component}")
    output = ROOT / "outputs" / f"{component}.onnx"
    structure = ROOT / "outputs" / f"{component}-structure.onnx"
    if sha256(WEIGHTS) != WEIGHTS_SHA256 or sha256(CONFIG) != CONFIG_SHA256:
        raise ValueError("Model weights or config do not match the pinned revision")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if not config["tie_word_embeddings"] or config["rope_scaling"] is not None:
        raise ValueError("Unexpected model configuration")

    with WEIGHTS.open("rb") as stream:
        header_size = struct.unpack("<Q", stream.read(8))[0]
        tensors = json.loads(stream.read(header_size))
    tensors.pop("__metadata__", None)
    data_start = 8 + header_size
    check_tied_embeddings(tensors, data_start)

    source = structure if structure.exists() else output
    model = onnx.load_model(str(source), load_external_data=False)
    graph = model.graph
    if graph.initializer:
        raise ValueError(f"Expected a weightless graph: {source}")
    graph_inputs = {value.name: value for value in graph.input}
    model_inputs = {
        name for name in graph_inputs if name.startswith("model.") or name in tensors
    }
    expected = set(tensors) - {"model.embed_tokens.weight"}
    if model_inputs != expected | {ROPE_NAME} or len(expected) != 310:
        raise ValueError("Graph weight inputs do not match the pinned model")

    for name in expected:
        value = graph_inputs[name]
        tensor = tensors[name]
        if tensor["dtype"] != "BF16":
            raise ValueError(f"Unexpected safetensors dtype: {name}")
        if value.type.tensor_type.elem_type != onnx.TensorProto.BFLOAT16:
            raise ValueError(f"Unexpected ONNX dtype: {name}")
        if value_shape(value) != tensor["shape"]:
            raise ValueError(f"Weight shape mismatch: {name}")

    rope_value = graph_inputs[ROPE_NAME]
    head_dim = config["head_dim"]
    if rope_value.type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
        raise ValueError("Unexpected RoPE dtype")
    if value_shape(rope_value) != [head_dim // 2]:
        raise ValueError("Unexpected RoPE shape")
    powers = np.arange(0, head_dim, 2, dtype=np.float32) / np.float32(head_dim)
    inv_freq = np.float32(1.0) / np.power(np.float32(config["rope_theta"]), powers)

    added = []
    with WEIGHTS.open("rb") as stream:
        for value in list(graph.input):
            name = value.name
            if name not in expected:
                continue
            tensor_info = tensors[name]
            start, end = tensor_info["data_offsets"]
            stream.seek(data_start + start)
            raw = stream.read(end - start)
            if len(raw) != end - start:
                raise ValueError(f"Truncated weight: {name}")
            tensor = onnx.TensorProto()
            tensor.name = name
            tensor.data_type = onnx.TensorProto.BFLOAT16
            tensor.dims.extend(tensor_info["shape"])
            tensor.raw_data = raw
            graph.initializer.append(tensor)
            added.append(name)

    rope = onnx.TensorProto()
    rope.name = ROPE_NAME
    rope.data_type = onnx.TensorProto.FLOAT
    rope.dims.extend([head_dim // 2])
    rope.raw_data = inv_freq.tobytes()
    graph.initializer.append(rope)
    added.append(ROPE_NAME)

    runtime_inputs = [value for value in graph.input if value.name not in model_inputs]
    expected_inputs = {"decode": 59, "prefill": 3}[component]
    if len(runtime_inputs) != expected_inputs or len(added) != 311:
        raise ValueError("Unexpected initializer or runtime input count")
    del graph.input[:]
    graph.input.extend(runtime_inputs)

    with tempfile.NamedTemporaryFile(dir=output.parent, prefix=f".{component}-weighted-", suffix=".onnx", delete=False) as temp:
        temp_path = Path(temp.name)
    try:
        onnx.save_model(model, str(temp_path))
        onnx.checker.check_model(str(temp_path))
        output_sha = sha256(temp_path)
        output_size = temp_path.stat().st_size

        old_sha = sha256(source)
        temp_path.chmod(source.stat().st_mode & 0o777)
        if not structure.exists():
            os.replace(output, structure)
        os.replace(temp_path, output)

        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        if manifest["revision"] != REVISION:
            raise ValueError("Manifest revision mismatch")
        record = manifest["components"][component]
        record.update({
            "file": output.name,
            "sha256": output_sha,
            "bytes": output_size,
            "inputs": [item for item in record["inputs"] if item["name"] not in model_inputs],
            "initializers": added,
            "checker": "passed",
            "weight_source": str(WEIGHTS.relative_to(ROOT)),
            "weight_source_sha256": WEIGHTS_SHA256,
            "weight_source_deleted_after_embedding": False,
            "structure_file": structure.name,
            "structure_sha256": old_sha,
        })
        with tempfile.NamedTemporaryFile(dir=MANIFEST.parent, prefix=".manifest-", suffix=".json", mode="w", encoding="utf-8", delete=False) as temp_manifest:
            json.dump(manifest, temp_manifest, indent=2, ensure_ascii=False)
            temp_manifest.write("\n")
            manifest_path = Path(temp_manifest.name)
        os.replace(manifest_path, MANIFEST)
        print(f"Weighted ONNX: {output} ({output_size} bytes, SHA-256 {output_sha})")
        print(f"Initializers: {len(added)}; runtime inputs: {len(runtime_inputs)}")
    finally:
        temp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
