import os, gc, json, time, signal, tempfile, pathlib, hashlib, importlib.metadata
signal.alarm(60)
import torch
from safetensors.torch import save_file
from fastsafetensors import SafeTensorsFileLoader, SingleGroup
torch.set_num_threads(1)
torch.cuda.set_per_process_memory_fraction((512*2**20)/torch.cuda.get_device_properties(0).total_memory)
assert hasattr(torch._C, "_host_emptyCache")
torch.cuda.init()
torch._C._host_emptyCache()
def stats(label):
    h=torch.cuda.host_memory_stats()
    m=dict((line.split(":")[0],int(line.split()[1])*1024) for line in pathlib.Path("/proc/meminfo").read_text().splitlines() if line.startswith(("MemAvailable:","Mlocked:","Unevictable:")))
    r={"phase":label,"host_owned":h.get("allocated_bytes.current"),"host_active":h.get("active_bytes.current"),"host_peak":h.get("allocated_bytes.peak"),"num_host_free":h.get("num_host_free"),"device_allocated":torch.cuda.memory_allocated(),"device_reserved":torch.cuda.memory_reserved(),**m}
    print(json.dumps(r),flush=True);return r
print(json.dumps({"torch":torch.__version__,"fastsafetensors":importlib.metadata.version("fastsafetensors"),"cuda":torch.version.cuda,"device":torch.cuda.get_device_name(0)}),flush=True)
stats("initial")
for mib in [32,65]:
    with tempfile.TemporaryDirectory(prefix="fst-hostcache-",dir="/tmp") as td:
        path=str(pathlib.Path(td)/"synthetic.safetensors")
        cpu=torch.full((mib*2**20,),37,dtype=torch.uint8)
        save_file({"synthetic":cpu},path);del cpu
        gc.collect()
        t0=time.perf_counter()
        loader=SafeTensorsFileLoader(SingleGroup(), "cuda:0", nogds=True, set_numa=False)
        loader.add_filenames({0:[path]})
        fb=loader.copy_files_to_device()
        t=fb.get_tensor("synthetic").clone()
        torch.cuda.synchronize()
        assert t.numel()==mib*2**20 and bool(torch.all(t==37).item())
        stats(str(mib)+"MiB_after_copy")
        fb.close();loader.close();del fb,loader
        torch.cuda.synchronize();gc.collect();torch.cuda.empty_cache()
        a=stats(str(mib)+"MiB_after_close_device_flush_tensor_live")
        torch._C._host_emptyCache()
        b=stats(str(mib)+"MiB_after_host_flush_tensor_live")
        assert bool(torch.all(t==37).item())
        del t
        torch.cuda.synchronize();gc.collect();torch.cuda.empty_cache();torch._C._host_emptyCache()
        c=stats(str(mib)+"MiB_final")
        print(json.dumps({"case_MiB":mib,"elapsed_seconds":time.perf_counter()-t0,"host_released":a["host_owned"]-b["host_owned"],"correct_before_after_host_release":True}),flush=True)
print(json.dumps({"result":"passed","remaining_tempdirs":len(list(pathlib.Path("/tmp").glob("fst-hostcache-*")))}),flush=True)
