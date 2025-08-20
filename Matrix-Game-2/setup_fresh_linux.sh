#!/usr/bin/env bash
# setup_fresh_linux.sh — One-shot setup for a clean Ubuntu server for Matrix-Game-2
# Safe-by-default with prompts disabled. Installs Miniconda (local), creates env,
# installs GPU deps, downloads model, ready to run inference.
# Re-run safe (idempotent-ish).

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_NAME="matrix-game-2.0"
CONDA_DIR="$HOME/miniconda"
HF_MODEL="Skywork/Matrix-Game-2.0"
MODEL_DIR="$PROJECT_DIR/pretrained_model"
PYTHON_VERSION="3.10"

log() { echo -e "\n[setup] $*\n"; }
warn() { echo -e "\n[setup][WARN] $*\n"; }

need_cmd() { command -v "$1" >/dev/null 2>&1; }

log "Project: $PROJECT_DIR"

# 0) Optional: print system info first
if [ -x "$PROJECT_DIR/check_system.sh" ]; then
  log "Running check_system.sh (info only)"
  bash "$PROJECT_DIR/check_system.sh" || true
else
  log "Tip: run ./check_system.sh for system info (optional)"
fi

# 1) Basic tools (skip if not using apt)
if need_cmd apt-get; then
  log "Updating apt and installing base tools"
  sudo apt-get update -y || true
  sudo apt-get install -y --no-install-recommends \
    git curl wget ca-certificates build-essential pkg-config \
    software-properties-common > /dev/null || true
else
  warn "apt-get not found; skipping base package installs"
fi

# 2) Miniconda install (local, no sudo)
if ! need_cmd conda; then
  if [ -d "$CONDA_DIR" ]; then
    warn "Miniconda directory exists at $CONDA_DIR but 'conda' not in PATH. Attempting upgrade install (-u)."
  else
    log "Installing Miniconda to $CONDA_DIR"
  fi
  mkdir -p "$HOME"
  cd "$HOME"
  curl -fsSL -o Miniconda3.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  if [ -d "$CONDA_DIR" ]; then
    bash Miniconda3.sh -u -b -p "$CONDA_DIR" || true
  else
    bash Miniconda3.sh -b -p "$CONDA_DIR"
  fi
  rm -f Miniconda3.sh
else
  log "Conda already present: $(conda --version || true)"
fi

# 3) Initialize conda for this shell (with fallback)
log "Initializing conda in current shell"
if [ -x "$CONDA_DIR/bin/conda" ]; then
  eval "$($CONDA_DIR/bin/conda shell.bash hook)" || true
fi
if ! need_cmd conda && [ -x "$CONDA_DIR/bin/conda" ]; then
  warn "'conda' still not on PATH; using absolute path for subsequent conda commands."
  alias conda="$CONDA_DIR/bin/conda"
fi

# 4) Accept ToS for default channels (handles non-interactive hosts)
log "Accepting Conda Terms of Service for Anaconda channels (if required)"
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true

# 5) Create env if missing
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  log "Conda env $ENV_NAME already exists"
else
  log "Creating conda env $ENV_NAME (python=$PYTHON_VERSION)"
  conda create -y -n "$ENV_NAME" python="$PYTHON_VERSION"
fi

log "Activating env $ENV_NAME"
conda activate "$ENV_NAME"

# 6) Pick proper PyTorch wheel based on CUDA from driver
CUDA_TAG="cpu"
if need_cmd nvidia-smi; then
  CUDA_RUNTIME=$(nvidia-smi | awk '/CUDA Version/ {print $NF; exit}') || true
  case "$CUDA_RUNTIME" in
    12.*) CUDA_TAG="cu128";;
    11.8*) CUDA_TAG="cu118";;
    11.7*) CUDA_TAG="cu117";;
    *) CUDA_TAG="cu121";;
  esac
fi

log "Installing PyTorch for tag: $CUDA_TAG"
if [ "$CUDA_TAG" = "cpu" ]; then
  pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
else
  pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/$CUDA_TAG
fi

# 7) Project dependencies
log "Installing project requirements"
pip install -U pip
pip install -r "$PROJECT_DIR/requirements.txt"

# 7b) GUI requirements (optional)
if [ -f "$PROJECT_DIR/gui-requirements.txt" ]; then
  log "Installing GUI requirements (gui-requirements.txt)"
  pip install -r "$PROJECT_DIR/gui-requirements.txt"
else
  warn "gui-requirements.txt not found; skipping GUI deps"
fi

# 8) Extra performance libs (best-effort)
log "Installing Flash-Attn (best-effort)"
pip install flash-attn==2.8.3 --no-build-isolation || warn "flash-attn install failed; continuing"

# 9) Hugging Face CLI and model download
log "Installing huggingface_hub"
pip install -U huggingface_hub
mkdir -p "$MODEL_DIR"
if need_cmd huggingface-cli; then
  log "Downloading model via huggingface-cli: $HF_MODEL -> $MODEL_DIR"
  huggingface-cli download "$HF_MODEL" --local-dir "$MODEL_DIR" || warn "HF download failed; check auth/token"
else
  log "huggingface-cli not found; using Python snapshot_download"
  python - <<'PY'
from huggingface_hub import snapshot_download
import os
model = os.environ.get('HF_MODEL', 'Skywork/Matrix-Game-2.0')
out = os.environ.get('MODEL_DIR')
snapshot_download(repo_id=model, local_dir=out, local_dir_use_symlinks=False)
print('Downloaded', model, 'to', out)
PY
fi

# 10) Quick GPU sanity check
log "Running sanity check"
python - <<'PY'
import torch
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device:', torch.cuda.get_device_name(0))
PY

eval "$($HOME/miniconda/bin/conda shell.bash hook)"
conda activate "$ENV_NAME"
python gui_app.py

log "Setup complete. Suggested next steps:"
# echo "  1) cd $PROJECT_DIR"
# echo "  2) conda activate $ENV_NAME"
# echo "  3) python inference.py --config_path configs/inference_yaml/inference_universal.yaml \\
#          --checkpoint_path pretrained_model/<checkpoint>.safetensors \\
#          --pretrained_model_path pretrained_model \\
#          --img_path demo_images/universal/0000.png \\
#          --output_folder outputs --seed 42"
