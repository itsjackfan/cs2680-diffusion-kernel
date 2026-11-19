#!/bin/bash
# Once per synced <exp> dir, on the cs2680 login node: bash cluster/setup.sh
# .venv = pyproject deps + the vLLM fork in vllm/ (editable, Python-only build: compiled ops come from the
# upstream v0.31.0 wheel, so no nvcc is needed). Rerun after changing C++/CUDA sources in vllm/csrc (needs a full
# build on a GPU node instead) or after a vLLM rebase; Python edits under vllm/vllm/ apply without rerunning.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
VLLM_WHEEL=$(python3 -c "import json,urllib.request as u; print(next(f['url'] for f in json.load(u.urlopen('https://pypi.org/pypi/vllm/0.31.0/json'))['urls'] if f['filename'].endswith('manylinux_2_28_x86_64.whl')))")
uv sync
VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_LOCATION="$VLLM_WHEEL" uv pip install -e ./vllm
uv run --no-sync python -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__)"
