# Qwen3-0.6B ONNX 导出与 PIM C 生成

本仓库保存 Qwen3-0.6B 的导出脚本、无权重 Prefill/Decode 结构图和导出清单。源模型是官方 `Qwen/Qwen3-0.6B`，固定 revision 为 `c1899de289a04d12100db370d81485cdf75e47ca`。带权重的 `prefill.onnx`、`decode.onnx` 由脚本生成，不提交到 Git。

## 给另一个 agent 的执行步骤

目标：在 WSL/Linux 中生成 `prefill.onnx`、`decode.onnx`，再用独立的 `pim_codegen` 仓库生成 `prefill.c`、`decode.c`。两个仓库放在同一父目录。若已有 `pim_codegen`，使用现有目录即可。

### 1. 克隆并生成 ONNX

```bash
cd ~/workspace
git clone https://github.com/heleadme/qwen3-0.6b-onnx-export.git
git clone https://github.com/heleadme/pim_codegen.git
cd qwen3-0.6b-onnx-export
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
python generate_prefill.py
python embed_decode_weights.py
sha256sum outputs/prefill.onnx outputs/decode.onnx
```

`generate_prefill.py` 下载固定 revision 的官方 `config.json` 和 `model.safetensors` 到 `model/`，核对 SHA-256 后，把权重嵌入仓库已有的 `outputs/prefill-structure.onnx`。第二条 Python 命令把相同权重嵌入 `outputs/decode-structure.onnx`。两条命令都会做 ONNX checker 校验。

预期 SHA-256：

```text
7fee5a71825ffa26ad48da49da2be1f8a15a40121dc025f49e713c990261ab91  outputs/prefill.onnx
04a3b297b01896db676f4c65fd7409fde3d17e73a0957ab02796c510e0e735e7  outputs/decode.onnx
```

下载和生成需要网络、足够内存及数 GB 磁盘空间。若校验值不一致，先记录文件大小、哈希及报错，不要继续生成 C。

### 2. 在 pim_codegen 中生成 C

`pim_codegen` 当前要求输入 ONNX 位于自身仓库内，因此先复制到它的 `models/` 子目录。下面的路径假设两个仓库是兄弟目录，且仍处于上一步激活的 Python 虚拟环境。

```bash
cd ../pim_codegen
mkdir -p models/qwen3-export
cp ../qwen3-0.6b-onnx-export/outputs/prefill.onnx models/qwen3-export/prefill.onnx
cp ../qwen3-0.6b-onnx-export/outputs/decode.onnx models/qwen3-export/decode.onnx

bash scripts/generate.sh \
  --model models/qwen3-export/prefill.onnx \
  --output output/qwen3-export/prefill.c

bash scripts/generate.sh \
  --model models/qwen3-export/decode.onnx \
  --output output/qwen3-export/decode.c \
  --weights-manifest output/qwen3-export/prefill.weights.json
```

第一条生成命令从 Prefill ONNX 自动打包 `prefill.bin` 和 `prefill.weights.json`，再生成 `prefill.c`。第二条命令复用这对权重文件生成 `decode.c`，避免再次打包约 1.2 GB 的相同权重；这里的 `--weights-manifest` 是生成 Decode 时指定已有权重清单所需。无需手动运行 `pack_qwen3_weights.py`，也不要添加 `--weight-layout`。

检查 `output/qwen3-export/` 中有 `prefill.c`、`decode.c`、`prefill.bin`、`prefill.weights.json`；两份 C 应包含 `test_qwen` 和 `kernel_3dpim`。若已存在同名输出，生成器会拒绝覆盖；改用新的输出目录，或先检查并处理旧产物。

当前本机已核对：本仓库生成的两份带权重 ONNX 的 SHA-256 与 `pim_codegen/models/Qwen3-0.6B/` 中的对应文件一致。C 生成属于生成器流程；它本身不证明目标端编译、链接或数值运行正确。若另一个版本的 `pim_codegen` 识别失败，请保留错误信息并核对仓库版本，不要直接手改生成的 C。

`export_structure.py` 可从固定 revision 重新导出无权重结构图；正常复现上述 C 时使用仓库已经保存的 `outputs/*-structure.onnx` 即可。上游模型许可证保存在 `model/LICENSE`。
