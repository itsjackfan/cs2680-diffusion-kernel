import json
from pathlib import Path

import pytest

from src.logger import Logger


@pytest.fixture
def logger(cfg, mocks):
    Path(cfg.run_dir).mkdir(parents=True)
    return Logger(cfg)


def test_init_naming(cfg, mocks, logger):
    kw = mocks.init.call_args.kwargs
    assert kw["name"] == "test-exp/test" and kw["group"] == "test-exp" and kw["dir"] == cfg.run_dir
    assert kw["mode"] == "disabled" and kw["resume"] == "allow" and kw["config"]["git"]["commit"] == "abc123"
    Logger(cfg)
    assert mocks.init.call_args.kwargs["id"] == kw["id"]


def test_rejects_unprefixed_keys(logger):
    with pytest.raises(ValueError):
        logger.log({"gen_ppl": 1.0}, step=0)


def test_scalars(cfg, mocks, logger):
    logger.log({"test/gen_ppl": 40.0, "perf/step_ms": 3.0}, step=7)
    assert mocks.log.call_args.args[0] == {"test/gen_ppl": 40.0, "perf/step_ms": 3.0}
    assert mocks.log.call_args.kwargs["step"] == 7
    lines = (Path(cfg.run_dir) / "metrics.jsonl").read_text().splitlines()
    assert json.loads(lines[-1]) == {"step": 7, "test/gen_ppl": 40.0, "perf/step_ms": 3.0}
