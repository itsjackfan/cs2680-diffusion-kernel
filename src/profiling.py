"""Per-region roofline of one denoising step: GPU time per labelled region (torch.profiler.record_function labels
in src/duo.py and the fork's duo.py) against an analytic cost model.

Cost model = ideal fused traffic: each region reads its inputs and bf16 weights once and writes its outputs once
(2 bytes per element; the posterior and categorical draw add no traffic). So intensity is an upper bound and
achieved GB/s / peak GB/s is the headroom a fused kernel could recover. Costs are per denoising step over n canvases.
Dims: n canvases, L canvas length, N = n * L tokens, D hidden, C cond dim, V vocab."""
import bisect
import json
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

REGIONS = ("adaln", "attention", "mlp", "output_proj", "sampling")
BYTES = 2


def region_costs(m, n):
    """-> {region: (flops, bytes)} for one step over n canvases; m = model config."""
    L, D, C, V, nb = m.length, m.hidden_dim, m.cond_dim, m.vocab_size, m.n_blocks
    N = n * L
    per_block = {
        "adaln": (2 * n * C * 6 * D + 20 * N * D, BYTES * (6 * C * D + 4 * N * D)),
        "attention": (2 * N * D * 3 * D + 4 * n * L * L * D + 2 * N * D * D, BYTES * (4 * D * D + 2 * N * D)),
        "mlp": (16 * N * D * D, BYTES * (8 * D * D + 2 * N * D)),
    }
    costs = {r: (nb * f, nb * b) for r, (f, b) in per_block.items()}
    costs["output_proj"] = (2 * N * D * V + 2 * n * C * 2 * D, BYTES * (D * V + N * D + N * V))
    costs["sampling"] = (20 * N * V, BYTES * N * V + 8 * N)
    return costs


def gpu_region_ms(trace_events):
    """Kernel time inside each region's GPU-side label window (chrome-trace gpu_user_annotation events).
    The CPU-side events' child-kernel totals double-counted kernels in vLLM at batch 8, so they are not used."""
    k = sorted((e["ts"], e["ts"] + e["dur"]) for e in trace_events if e.get("cat") == "kernel")
    starts = [a for a, _ in k]
    ms = dict.fromkeys(REGIONS, 0.0)
    for e in trace_events:
        if e.get("cat") == "gpu_user_annotation" and e["name"] in ms:
            a, b = e["ts"], e["ts"] + e["dur"]
            i = bisect.bisect_left(starts, a)
            while i < len(k) and k[i][0] < b:
                ms[e["name"]] += (min(k[i][1], b) - k[i][0]) / 1e3
                i += 1
    return ms


def region_times(fn, device, trace):
    """Run fn under the profiler, save the chrome trace -> {region: total ms} (GPU kernel time on cuda,
    CPU time of the label events on cpu)."""
    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if device == "cuda" else [])
    with profile(activities=acts) as prof:
        fn()
    prof.export_chrome_trace(trace)
    if device == "cuda":
        return gpu_region_ms(json.loads(Path(trace).read_text())["traceEvents"])
    ms = dict.fromkeys(REGIONS, 0.0)
    for e in prof.events():
        if e.name in ms:
            ms[e.name] += e.cpu_time_total / 1e3
    return ms


def _ms(fn, iters=10):
    fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return 1e3 * (time.perf_counter() - t0) / iters


def peaks(device):
    """Measured (bf16 matmul TFLOP/s, device copy GB/s) of this GPU."""
    a = torch.randn(8192, 8192, device=device, dtype=torch.bfloat16)
    x = torch.empty(2**28, device=device, dtype=torch.float32)
    tflops = 2 * 8192**3 / _ms(lambda: a @ a) / 1e9
    gbps = 2 * x.numel() * 4 / _ms(lambda: x.clone()) / 1e6
    return tflops, gbps


def metrics(m, n, region_ms, step_ms, peak_tflops, peak_gbps):
    """region_ms: per-step ms per region -> perf/<region>/{ms, tflops, gbps, intensity, memory_bound} + peaks."""
    out = {"perf/peak_tflops": peak_tflops, "perf/peak_gbps": peak_gbps,
           "perf/overhead_ms": step_ms - sum(region_ms.values())}
    for r, (f, b) in region_costs(m, n).items():
        ms = region_ms[r]
        out |= {f"perf/{r}/ms": ms, f"perf/{r}/tflops": f / ms / 1e9, f"perf/{r}/gbps": b / ms / 1e6,
                f"perf/{r}/intensity": f / b, f"perf/{r}/memory_bound": f / b < 1e3 * peak_tflops / peak_gbps}
    return out
