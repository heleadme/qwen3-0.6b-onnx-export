"""Download pinned official weights and generate outputs/prefill.onnx."""

from huggingface_hub import hf_hub_download

from embed_decode_weights import REVISION, ROOT, main


def generate_prefill():
    structure = ROOT / "outputs" / "prefill-structure.onnx"
    if not structure.is_file():
        raise FileNotFoundError(f"Missing weightless ONNX graph: {structure}")
    for filename in ("config.json", "model.safetensors"):
        hf_hub_download(
            repo_id="Qwen/Qwen3-0.6B",
            filename=filename,
            revision=REVISION,
            local_dir=str(ROOT / "model"),
        )
    main("prefill")


if __name__ == "__main__":
    generate_prefill()
