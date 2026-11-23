"""Write the pinned Qwen3-0.6B weights into outputs/prefill.onnx."""

from embed_decode_weights import main


if __name__ == "__main__":
    main("prefill")
