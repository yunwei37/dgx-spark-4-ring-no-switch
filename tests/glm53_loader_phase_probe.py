import datetime,gc,hashlib,json,pathlib,re,threading,time
T0=time.monotonic()
def kb(path):
 out={}
 for line in pathlib.Path(path).read_text().splitlines():
  k,_,v=line.partition(":")
  if v.strip() and v.strip().split()[0].isdigit(): out[k]=int(v.strip().split()[0])*1024
 return out
def snap(phase,**extra):
 m=kb("/proc/meminfo");s=kb("/proc/self/status")
 assert m["MemAvailable"]>=64*1024**3,"64GiB host headroom"
 z=pathlib.Path("/proc/zoneinfo").read_text().split("zone   Normal",1)[-1]
 n=re.search(r"pages free\s+(\d+)",z)
 r={"phase":phase,"seconds":time.monotonic()-T0,"utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"MemAvailable":m["MemAvailable"],"Normal_free_pages":int(n.group(1)) if n else None,"process":{k:s.get(k) for k in ["VmRSS","RssAnon","RssFile","VmLck","VmPin"]},**extra}
 if "torch" in globals() and torch.cuda.is_initialized():
  r["cuda"]={"allocated":torch.cuda.memory_allocated(),"reserved":torch.cuda.memory_reserved(),"peak":torch.cuda.max_memory_allocated()}
  if hasattr(torch.cuda.memory,"host_memory_stats"):
   try: r["host_allocator"]=dict(torch.cuda.memory.host_memory_stats())
   except Exception as e: r["host_allocator_unavailable"]=type(e).__name__
 print(json.dumps(r),flush=True)
snap("entry")
import torch
snap("torch_import",torch_version=torch.__version__)
import safetensors
snap("safetensors_import",safetensors_version=safetensors.__version__)
root=pathlib.Path("/model/LibertAIDAI--GLM-5.3-NVFP4")
assert json.loads((root/".spark-manager-verified").read_text())["revision"]=="653d98214a487d78c1b2a37167b56764b6240080"
path=root/"model-00001-of-00282.safetensors"
with path.open("rb") as f:
 size=int.from_bytes(f.read(8),"little"); assert size<16*1024**2
 header=json.loads(f.read(size))
entries=[(v["data_offsets"][1]-v["data_offsets"][0],k,v) for k,v in header.items() if k!="__metadata__"]
tensor_bytes,name,meta=sorted(entries,key=lambda x:(-x[0],x[1]))[0]
assert tensor_bytes==1903165440
snap("header",file=path.name,file_bytes=path.stat().st_size,name=name,tensor_bytes=tensor_bytes,dtype=meta["dtype"],shape=meta["shape"])
with safetensors.safe_open(str(path),framework="pt",device="cpu") as f:
 snap("safe_open")
 tensor=f.get_tensor(name)
 pointer=tensor.data_ptr()
 mapping=None
 for line in pathlib.Path("/proc/self/maps").read_text().splitlines():
  parts=line.split();lo,hi=(int(x,16) for x in parts[0].split("-"))
  if lo<=pointer<hi:
   mapping={"bytes":hi-lo,"permissions":parts[1],"file":pathlib.Path(parts[-1]).name if len(parts)>=6 else "anonymous"};break
 snap("get_tensor",tensor_mapping=mapping)
 torch.cuda.init()
 assert torch.cuda.get_device_name(0)=="NVIDIA GB10"
 torch.cuda.set_per_process_memory_fraction(4*1024**3/torch.cuda.mem_get_info()[1])
 snap("cuda_init")
 destination=torch.empty_like(tensor,device="cuda")
 torch.cuda.synchronize()
 snap("destination_allocated")
 destination.copy_(tensor)
 torch.cuda.synchronize()
 snap("inplace_copy")
 cpu=tensor.reshape(-1).view(torch.uint8)
 gpu=destination.reshape(-1).view(torch.uint8)
 actual=bytes(gpu[:16].cpu().tolist()+gpu[-16:].cpu().tolist())
 expected=bytes(cpu[:16].tolist()+cpu[-16:].tolist())
 assert actual==expected
 snap("edges_equal",edge_sha256=hashlib.sha256(actual).hexdigest())
 del cpu,gpu,tensor
 gc.collect()
 snap("input_reference_deleted")
snap("file_context_closed")
del destination
gc.collect()
snap("destination_deleted")
torch.cuda.empty_cache()
snap("device_cache_released")
print(json.dumps({"phase":"diagnostic_pass","full_model":False,"inference_requests":0,"no_host_tuning":True}),flush=True)

