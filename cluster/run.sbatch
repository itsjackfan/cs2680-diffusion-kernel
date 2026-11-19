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
D=$(dirname "$1")
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$D/gpu.txt"
uv pip freeze > "$D/freeze.txt"
cp .gitinfo .gitinfo-vllm "$D/"  # repo + vLLM fork commits, written by cluster/hooks/post-commit
uv run --no-sync python -m src.run "$1"  # --no-sync: keeps the editable vLLM fork install from cluster/setup.sh
