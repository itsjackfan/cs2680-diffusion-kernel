"""Per-region profiling: analytic cost model, labelled-region timing, profiled run. CPU-only, WandB mocked."""
import json
from pathlib import Path

from omegaconf import open_dict

from src import duo, profiling
from src.config import compose
from src.run import main
from tests.conftest import EOS, TINY, logged, tiny_dit


def test_profile_eval_config():
    ev = compose(["eval=profile"]).eval
    assert ev.profile_steps > 0 and ev.n_samples > 0 and ev.judge == "gpt2-large"


def test_region_costs(cfg):
    costs = profiling.region_costs(cfg.model, n=2)
    assert set(costs) == set(profiling.REGIONS)
    assert all(f > 0 and b > 0 for f, b in costs.values())
    flops2, _ = costs["mlp"]
    assert profiling.region_costs(cfg.model, n=4)["mlp"][0] == 2 * flops2  # linear in canvases


def test_region_times_cover_all_regions(tmp_path):
    dit = tiny_dit()
    ms = profiling.region_times(lambda: duo.sample(dit, 2, 8, TINY["vocab_size"], 2, EOS), "cpu",
                                str(tmp_path / "trace.json"))
    assert set(ms) == set(profiling.REGIONS) and all(v > 0 for v in ms.values())


def test_gpu_region_ms_counts_kernel_time_inside_windows():
    ev = [{"cat": "gpu_user_annotation", "name": "sampling", "ts": 0, "dur": 100},
          {"cat": "gpu_user_annotation", "name": "attention", "ts": 200, "dur": 50},
          {"cat": "kernel", "name": "k", "ts": 10, "dur": 20},    # inside sampling: 20 us
          {"cat": "kernel", "name": "k", "ts": 90, "dur": 30},    # straddles the end: 10 us
          {"cat": "kernel", "name": "k", "ts": 150, "dur": 10},   # between windows: unlabelled
          {"cat": "kernel", "name": "k", "ts": 200, "dur": 10}]   # inside attention: 10 us
    ms = profiling.gpu_region_ms(ev)
    assert ms["sampling"] == 0.03 and ms["attention"] == 0.01 and ms["mlp"] == 0.0


def test_run_logs_region_metrics(cfg, mocks, monkeypatch):
    monkeypatch.setattr("src.profiling.peaks", lambda device: (100.0, 500.0))
    with open_dict(cfg):
        cfg.eval.profile_steps = 2
    main(cfg)
    ((metrics, _),) = logged(mocks)
    for r in profiling.REGIONS:
        assert {f"perf/{r}/{k}" for k in ("ms", "tflops", "gbps", "intensity", "memory_bound")} <= set(metrics)
    assert metrics["perf/peak_tflops"] == 100.0 and "perf/overhead_ms" in metrics and "test/gen_ppl" in metrics
    assert json.loads((Path(cfg.run_dir) / "trace.json").read_text())
