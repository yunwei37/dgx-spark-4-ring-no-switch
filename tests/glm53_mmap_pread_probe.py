import ast,collections,concurrent.futures,hashlib,itertools,json,os,pathlib,sys,threading,time
from typing import List,Tuple,Generator
import torch,safetensors,safetensors.torch
from tqdm.auto import tqdm
MODE=sys.argv[1]
RUN=sys.argv[2]
import functools,inspect,types
assert MODE in ['mmap','pread']
assert safetensors.__version__=='0.8.0'
api_signature=str(inspect.signature(safetensors.safe_open))
assert 'backend' in inspect.signature(safetensors.safe_open).parameters
native_safetensors=safetensors
safetensors=types.SimpleNamespace(safe_open=functools.partial(native_safetensors.safe_open,backend=MODE),torch=native_safetensors.torch)
ROOT=pathlib.Path("/model/LibertAIDAI--GLM-5.3-NVFP4")
assert json.loads((ROOT/".spark-manager-verified").read_text())["revision"]=="653d98214a487d78c1b2a37167b56764b6240080"
files=sorted(set(json.loads((ROOT/"model.safetensors.index.json").read_text())["weight_map"].values()))[:10]
paths=[str(ROOT/f) for f in files]
source=pathlib.Path("/sgl-workspace/sglang/python/sglang/srt/model_loader/weight_utils.py").read_bytes()
assert hashlib.sha256(source).hexdigest()=="d82dc59e8d4a2fafac2e61c468da485e9f7a85042cf9044b6b37c3a3b6b86041"
tree=ast.parse(source)
wanted={"safetensors_weights_iterator","buffered_multi_thread_safetensors_weights_iterator"}
nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in wanted]
assert len(nodes)==2
BAR_FORMAT="{l_bar}{bar}| {n_fmt}/{total_fmt}"
exec(compile(ast.Module(body=nodes,type_ignores=[]),"<pinned native iterators>","exec"),globals())
torch.cuda.init()
assert torch.cuda.get_device_name(0)=="NVIDIA GB10"
torch.cuda.set_per_process_memory_fraction(4*1024**3/torch.cuda.mem_get_info()[1])
def numbers(path):
 out={}
 for l in pathlib.Path(path).read_text().splitlines():
  k,_,v=l.partition(":")
  if v.strip() and v.strip().split()[0].isdigit():out[k]=int(v.strip().split()[0])*1024
 return out
baseline=numbers("/proc/meminfo")
assert baseline["MemAvailable"]>=64*1024**3
samples=[]
stop=threading.Event()
low=threading.Event()
def sample():
 while not stop.is_set():
  m=numbers("/proc/meminfo");s=numbers("/proc/self/status")
  z=pathlib.Path("/proc/zoneinfo").read_text()
  normal=z.split("zone   Normal",1)[-1] if "zone   Normal" in z else ""
  import re
  match=re.search(r"pages free\s+(\d+)",normal)
  mins=re.search(r"\n\s+min\s+(\d+)",normal)
  samples.append({"t":time.monotonic(),"MemAvailable":m.get("MemAvailable"),
   "RssAnon":s.get("RssAnon"),"RssFile":s.get("RssFile"),"VmRSS":s.get("VmRSS"),
   "VmLck":s.get("VmLck"),"VmPin":s.get("VmPin"),
   "Normal_free_pages":int(match.group(1)) if match else None,
   "Normal_min_pages":int(mins.group(1)) if mins else None})
  if m["MemAvailable"]<64*1024**3:low.set()
  stop.wait(.25)
thread=threading.Thread(target=sample)
thread.start()
start=time.monotonic();last=start;count=0;total=0;digest=hashlib.sha256()
print(json.dumps({"phase":"start","mode":MODE,"run":RUN,"source_sha256":hashlib.sha256(source).hexdigest(),"files":files,"file_bytes":sum(pathlib.Path(f).stat().st_size for f in paths),"backend":MODE,"safe_open_signature":api_signature,"torch_version":torch.__version__,"safetensors_version":native_safetensors.__version__,"prefetch":False,"drop_cache":False,"cuda_cap":4*1024**3}),flush=True)
iterator=buffered_multi_thread_safetensors_weights_iterator(paths,max_workers=8)
try:
 for name,param in iterator:
  assert not low.is_set(),"64GiB host headroom boundary"
  g=param.to("cuda")
  torch.cuda.synchronize()
  raw=g.contiguous().reshape(-1).view(torch.uint8)
  digest.update(name.encode())
  digest.update(raw[:16].cpu().numpy().tobytes())
  digest.update(raw[-16:].cpu().numpy().tobytes())
  count+=1;total+=g.numel()*g.element_size()
  del raw,g,param
  if time.monotonic()-last>15:
   print(json.dumps({"phase":"progress","mode":MODE,"run":RUN,"tensors":count,"bytes":total,"seconds":time.monotonic()-start}),flush=True);last=time.monotonic()
finally:
 stop.set();thread.join(timeout=2)
r={"phase":"result","mode":MODE,"run":RUN,"seconds":time.monotonic()-start,"tensors":count,"tensor_bytes":total,"edge_digest":digest.hexdigest(),"source_sha256":hashlib.sha256(source).hexdigest(),"cuda_peak_allocated":torch.cuda.max_memory_allocated(),"cuda_peak_reserved":torch.cuda.max_memory_reserved(),"mem_available_start":baseline["MemAvailable"],"mem_available_min":min(x["MemAvailable"] for x in samples),"process_peaks":{k:max(x[k] for x in samples if x[k] is not None) for k in ["VmRSS","RssAnon","RssFile","VmLck","VmPin"]},"normal_free_pages_min":min((x["Normal_free_pages"] for x in samples if x["Normal_free_pages"] is not None),default=None),"samples":len(samples),"scope":"real first-ten-shard CPU-to-GPU copies, not complete model or inference; cache order uncontrolled"}
print(json.dumps(r),flush=True)
pathlib.Path("/tmp/"+MODE+"-"+RUN+".json").write_text(json.dumps(r))
