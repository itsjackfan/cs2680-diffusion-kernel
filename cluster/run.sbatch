#!/bin/bash
# One run. Submitted only by cluster/launch.py (cwd = synced <exp> dir): sbatch cluster/run.sbatch <run_dir>/config.yaml
# gpu-cs2680-long = 1x RTX PRO 6000 (96 GB, sm_120), 8 CPUs, 62 GB RAM, 7-day limit.
#SBATCH --partition=gpu-cs2680-long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=56G
#SBATCH --time=06:00:00
#SBATCH --open-mode=append
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
set -a; source .env; set +a
export PATH="$HOME/.local/bin:$PATH" VLLM_ENABLE_V1_MULTIPROCESSING=0  # in-process engine: src.run times it directly
# venv: built by the first job of each synced <exp> dir (the login node is too small for uv); flock serialises sweeps.
# The vLLM fork is editable (Python edits apply live) with compiled ops from the upstream wheel. After a fork rebase
# or a vllm/csrc change, delete .venv to force a reinstall.
WHEEL=https://files.pythonhosted.org/packages/e1/98/841d1328827dc082492fcbb0f57bdf6a9d35aa0df52bd47b75be3f504804/vllm-0.31.0-cp38-abi3-manylinux_2_28_x86_64.whl
(
  flock 9
  uv sync --inexact  # --inexact keeps the fork, which pyproject does not list
  uv run --no-sync python -c "import vllm" 2>/dev/null \
    || VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_LOCATION=$WHEEL uv pip install -e ./vllm
) 9> .venv.lock
D=$(dirname "$1")
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$D/gpu.txt"
uv pip freeze > "$D/freeze.txt"
cp .gitinfo .gitinfo-vllm "$D/"  # repo + vLLM fork commits, written by cluster/hooks/post-commit
uv run --no-sync python -m src.run "$1"  # --no-sync: keeps the editable vLLM fork install
