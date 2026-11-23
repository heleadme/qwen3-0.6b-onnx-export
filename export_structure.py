import os, json, hashlib, sys, traceback
from pathlib import Path
import torch, onnx, transformers
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.exporters import OnnxExporter, OnnxConfig
from huggingface_hub import hf_hub_download
root=Path(__file__).resolve().parent
out=root/'outputs'; out.mkdir(exist_ok=True)
if (out/'decode-structure.onnx').exists() and (out/'decode.onnx').exists():
 raise RuntimeError('Refusing to overwrite the weighted outputs/decode.onnx')
sha='c1899de289a04d12100db370d81485cdf75e47ca'
path=hf_hub_download('Qwen/Qwen3-0.6B','config.json',revision=sha,local_dir=str(root/'official-config'))
data=Path(path).read_bytes(); raw=json.loads(data)
expected={'num_hidden_layers':28,'hidden_size':1024,'intermediate_size':3072,'num_attention_heads':16,'num_key_value_heads':8,'head_dim':128,'vocab_size':151936,'max_position_embeddings':40960,'torch_dtype':'bfloat16','architectures':['Qwen3ForCausalLM']}
assert all(raw.get(k)==v for k,v in expected.items()), 'official config mismatch'
config=AutoConfig.from_pretrained('Qwen/Qwen3-0.6B',revision=sha)
print('revision',sha,'config sha256',hashlib.sha256(data).hexdigest(),flush=True)
model=AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16).eval()
print('parameters',sum(p.numel() for p in model.parameters()),'bytes',sum(p.numel()*p.element_size() for p in model.parameters()),'dtypes',set(str(p.dtype) for p in model.parameters()),flush=True)
inputs={'input_ids':torch.arange(64,dtype=torch.long).reshape(1,64),'attention_mask':torch.ones((1,64),dtype=torch.long)}
programs=OnnxExporter().export_for_generation(model,inputs,OnnxConfig(export_params=False,optimize=False,dynamic=False,output_path=None))
print('components',list(programs),flush=True)
manifest={'revision':sha,'config_sha256':hashlib.sha256(data).hexdigest(),'torch':torch.__version__,'transformers':transformers.__version__,'onnx':onnx.__version__,'components':{}}
for name,program in programs.items():
 p=out/(name+'.onnx'); program.save(p,include_initializers=False,keep_initializers_as_inputs=True)
 onnx.checker.check_model(str(p)); graph=onnx.load(str(p),load_external_data=False).graph
 def desc(v):
  t=v.type.tensor_type
  return {'name':v.name,'dtype':onnx.TensorProto.DataType.Name(t.elem_type),'shape':[d.dim_value if d.HasField('dim_value') else d.dim_param for d in t.shape.dim]}
 from collections import Counter
 manifest['components'][name]={'file':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size,'inputs':[desc(v) for v in graph.input],'outputs':[desc(v) for v in graph.output],'initializers':[x.name for x in graph.initializer],'node_counts':dict(Counter(x.op_type for x in graph.node)),'checker':'passed'}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
print('manifest saved',flush=True)
