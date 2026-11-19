"""End-to-end smoke of one run (HF engine, tiny Duo, tiny judge, WandB mocked)."""
import json
from pathlib import Path

from src.run import main
from tests.conftest import logged

KEYS = {"test/gen_ppl", "test/entropy", "perf/samples_per_s", "perf/tokens_per_s", "perf/latency_s", "perf/step_ms",
        "perf/peak_mem_gb"}


def test_run_logs_quality_and_perf(cfg, mocks):
    main(cfg)
    ((metrics, step),) = logged(mocks)
    assert set(metrics) == KEYS and step == 0
    run_dir = Path(cfg.run_dir)
    samples = [json.loads(line) for line in (run_dir / "samples.jsonl").read_text().splitlines()]
    assert len(samples) == cfg.eval.n_samples and len(samples[0]["tokens"]) == cfg.model.length
    assert (run_dir / "config.yaml").exists() and (run_dir / "metrics.jsonl").exists()
