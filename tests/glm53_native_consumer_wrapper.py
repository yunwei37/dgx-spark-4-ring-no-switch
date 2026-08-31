# Bounded diagnostic only: retain native one-layer construction and real loading.
import os as _probe_os
import pathlib as _probe_path
import hashlib as _probe_hash
import threading as _probe_thread
import time as _probe_time
import json as _probe_json
import re as _probe_re
from unittest.mock import patch as _probe_patch

_probe_mode = _probe_os.environ["RING_CONSUMER_MODE"]
assert _probe_mode in ("async", "sync")
_probe_root = _probe_path.Path("/model/LibertAIDAI--GLM-5.3-NVFP4")
_probe_config = _probe_json.loads((_probe_root / "config.json").read_text())
_probe_layer = int(_probe_config["first_k_dense_replace"])
assert _probe_config["num_hidden_layers"] == 78
assert _probe_config["n_routed_experts"] == 256
assert 0 <= _probe_layer < 78
_probe_samples = []
_probe_stop = _probe_thread.Event()
_probe_low = _probe_thread.Event()
_probe_stats = {}
_probe_init = _initialize_model
_probe_post = DefaultModelLoader.load_weights_and_postprocess

def _probe_mem():
    def read(p):
        d={}
        for line in _probe_path.Path(p).read_text().splitlines():
            k,_,v=line.partition(":")
            if v.strip() and v.split()[0].isdigit():d[k]=int(v.split()[0])*1024
        return d
    m=read("/proc/meminfo"); p=read("/proc/self/status")
    return {k:p.get(k,0) for k in ("VmRSS","RssAnon","RssFile","VmPin","VmLck")} | {"MemAvailable":m["MemAvailable"]}

def _probe_monitor():
    while not _probe_stop.is_set():
        s=_probe_mem(); _probe_samples.append(s)
        if s["MemAvailable"] < 64*1024**3:_probe_low.set()
        _probe_stop.wait(.05)

def _initialize_model(*args, **kwargs):
    import sglang.srt.models.deepseek_v2 as dv
    assert _probe_hash.sha256(_probe_path.Path(dv.__file__).read_bytes()).hexdigest() == "c78b4ad75ab2478f1911136da18c09d3353d8ffaaf32d0e701536d1c9234898c"
    assert _probe_mem()["MemAvailable"] >= 64*1024**3
    torch.cuda.set_per_process_memory_fraction(20*1024**3/torch.cuda.mem_get_info()[1])
    _probe_stats["started"]=_probe_time.monotonic()
    _probe_stats["mode"]=_probe_mode
    _probe_stats["layer_id"]=_probe_layer
    _probe_stats["full_model"]=False
    _probe_stats["inference_requests"]=0
    _probe_stats["samples_thread"]=_probe_thread.Thread(target=_probe_monitor,daemon=True)
    _probe_stats["samples_thread"].start()
    def component_layers(num_layers, layer_fn, **kw):
        assert num_layers == 78 and kw["pp_size"] == 1
        from sglang.srt.layers.utils import PPMissingLayer
        layers=torch.nn.ModuleList([
            layer_fn(i, f'{kw["prefix"]}.{i}') if i == _probe_layer else PPMissingLayer()
            for i in range(num_layers)
        ])
        return layers,_probe_layer,_probe_layer+1
    with _probe_patch.object(dv,"make_layers",component_layers):
        model=_probe_init(*args,**kwargs)
    assert len(model.model.layers)==78
    assert model.model.start_layer == _probe_layer and model.model.end_layer == _probe_layer+1
    _probe_stats["constructor_allocated"]=torch.cuda.memory_allocated()
    _probe_stats["constructor_host_available"]=_probe_mem()["MemAvailable"]
    print("GLM53_COMPONENT_CONSTRUCTOR="+_probe_json.dumps({k:v for k,v in _probe_stats.items() if k not in ("samples_thread",)}),flush=True)
    return model

def _probe_weights(self, model_config, model):
    from safetensors import safe_open
    index=_probe_json.loads((_probe_root/"model.safetensors.index.json").read_text())["weight_map"]
    wanted={n:f for n,f in index.items() if n.startswith(f"model.layers.{_probe_layer}.")
        or n in ("model.embed_tokens.weight","model.norm.weight","lm_head.weight")}
    experts={int(m.group(1)) for n in wanted if (m:=_probe_re.search(r"\.mlp\.experts\.(\d+)\.",n))}
    assert experts == set(range(256)),len(experts)
    assert all(n in wanted for n in ("model.embed_tokens.weight","model.norm.weight","lm_head.weight"))
    _probe_stats["selected_tensors"]=len(wanted)
    _probe_stats["selected_experts"]=len(experts)
    _probe_stats["selected_tensor_bytes"]=0
    _probe_stats["read_tensors"]=0
    _probe_stats["input_edge"]=_probe_hash.sha256()
    for filename in sorted(set(wanted.values())):
        assert not _probe_low.is_set(),"64GiB host reserve reached"
        with safe_open(str(_probe_root/filename),framework="pt",device="cpu",backend="pread") as f:
            for name in f.keys():
                if name not in wanted:continue
                assert not _probe_low.is_set(),"64GiB host reserve reached"
                w=f.get_tensor(name)
                edge=w.reshape(-1).view(torch.uint8)
                _probe_stats["input_edge"].update(name.encode())
                _probe_stats["input_edge"].update(edge[:16].numpy().tobytes())
                _probe_stats["input_edge"].update(edge[-16:].numpy().tobytes())
                del edge
                _probe_stats["read_tensors"]+=1
                _probe_stats["selected_tensor_bytes"]+=w.numel()*w.element_size()
                yield name,w
                del w
    assert _probe_stats["read_tensors"] == len(wanted)

def _probe_postprocess(model, weights, target_device):
    import sglang.srt.models.deepseek_common.deepseek_weight_loader as consumer
    assert _probe_hash.sha256(_probe_path.Path(consumer.__file__).read_bytes()).hexdigest()=="51ea4fe580ce161d60dd42d3c3bcf047cdcf70b311ddb9a660f0fa3c81d112ac"
    started=_probe_time.monotonic()
    pending={"tasks":0,"logical_bytes":0,"tasks_peak":0,"logical_bytes_peak":0,"submitted":0}
    lock=_probe_thread.Lock()
    native_submit=consumer.maybe_executor_submit
    def observed_submit(**kw):
        if not kw["use_async"]:
            return native_submit(**kw)
        weight=kw["func_args"][1]
        size=weight.numel()*weight.element_size()
        with lock:
            pending["tasks"]+=1;pending["logical_bytes"]+=size;pending["submitted"]+=1
            pending["tasks_peak"]=max(pending["tasks_peak"],pending["tasks"])
            pending["logical_bytes_peak"]=max(pending["logical_bytes_peak"],pending["logical_bytes"])
        def finished(_future):
            with lock:
                pending["tasks"]-=1;pending["logical_bytes"]-=size
        try:
            native_submit(**kw)
        except BaseException:
            finished(None);raise
        kw["futures"][-1].add_done_callback(finished)
    with _probe_patch.object(consumer,"maybe_executor_submit",observed_submit):
        if _probe_mode == "sync":
            with _probe_patch.object(consumer,"should_async_load",lambda weight:False):
                _probe_post(model,weights,target_device)
        else:
            _probe_post(model,weights,target_device)
    torch.cuda.synchronize()
    _probe_stats["pending_tasks"]=pending
    assert pending["tasks"]==0 and pending["logical_bytes"]==0
    _probe_stats["native_load_and_postprocess_seconds"]=_probe_time.monotonic()-started
    _probe_stats["peak_cuda_allocated"]=torch.cuda.max_memory_allocated()
    _probe_stats["peak_cuda_reserved"]=torch.cuda.max_memory_reserved()
    _probe_stats["input_edge_digest"]=_probe_stats.pop("input_edge").hexdigest()
    load_samples=list(_probe_samples)
    _probe_stats["load_process_peaks"]={k:max(x[k] for x in load_samples) for k in ("VmRSS","RssAnon","RssFile")}
    _probe_stats["load_host_available_min"]=min(x["MemAvailable"] for x in load_samples)
    dig=_probe_hash.sha256(); nbytes=0; count=0
    for name,p in sorted(model.named_parameters()):
        # Full parameter-byte equality after native transforms, bounded host chunks.
        raw=p.detach().contiguous().reshape(-1).view(torch.uint8)
        dig.update(name.encode());dig.update(str((tuple(p.shape),p.dtype)).encode())
        for off in range(0,raw.numel(),1024**2):
            assert not _probe_low.is_set(),"64GiB host reserve reached"
            dig.update(raw[off:off+1024**2].cpu().numpy().tobytes())
        nbytes+=raw.numel();count+=1
        del raw
    torch.cuda.synchronize()
    _probe_stop.set();_probe_stats.pop("samples_thread").join(timeout=2)
    _probe_stats["parameter_full_sha256"]=dig.hexdigest()
    _probe_stats["parameter_bytes"]=nbytes
    _probe_stats["parameter_count"]=count
    _probe_stats["process_peaks"]={k:max(x[k] for x in _probe_samples) for k in ("VmRSS","RssAnon","RssFile","VmPin","VmLck")}
    _probe_stats["host_available_min"]=min(x["MemAvailable"] for x in _probe_samples)
    _probe_stats["samples"]=len(_probe_samples)
    _probe_stats["total_seconds"]=_probe_time.monotonic()-_probe_stats.pop("started")
    _probe_stats["probe_scope"]="one real MoE layer/all256experts plus embedding/head/norm; not full architecture inference"
    print("GLM53_COMPONENT_RESULT="+_probe_json.dumps(_probe_stats,sort_keys=True),flush=True)
    raise RuntimeError("GLM53_COMPONENT_FINISHED_NOT_INFERENCE")

DefaultModelLoader._get_all_weights=_probe_weights
DefaultModelLoader.load_weights_and_postprocess=staticmethod(_probe_postprocess)
