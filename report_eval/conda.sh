#!/usr/bin/env bash
set -euo pipefail

# ── configuration ────────────────────────────────────────────────────
CONDA_DIR="$HOME/miniconda3"
ENV_NAME="avail"
MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
CONDA_BIN="$CONDA_DIR/bin/conda"
ACTION="${1:-create}"                # create | delete | activate
PKGS="pypy pandas numpy scipy matplotlib jupyterlab networkx pytz tzdata pygraphviz"

# ── helper: ensure Miniconda is available ────────────────────────────
install_miniconda() {
  if [[ ! -d $CONDA_DIR ]]; then
    echo "[+] Installing Miniconda into $CONDA_DIR ..."
    wget -q --show-progress "$MINICONDA_URL" -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$CONDA_DIR"
    rm /tmp/miniconda.sh
    echo "[✓] Miniconda installed."
  fi
  "$CONDA_BIN" init "$(basename "$SHELL")" >/dev/null 2>&1 || true
  "$CONDA_BIN" config --add channels conda-forge 2>/dev/null || true
  "$CONDA_BIN" config --set channel_priority flexible
}

# ── helper: activate env in a subshell ───────────────────────────────
activate_env() {
  exec bash --rcfile <(echo "source ~/.bashrc && conda activate $ENV_NAME") -i
}

# ── delete ───────────────────────────────────────────────────────────
if [[ $ACTION == delete ]]; then
  if [[ -x $CONDA_BIN ]]; then
    "$CONDA_BIN" env remove -n "$ENV_NAME" -y || true
    echo "[✓] Environment '$ENV_NAME' removed."
  else
    echo "[!] Conda not found – nothing to delete."
  fi
  exit 0
fi

# ── create / refresh ─────────────────────────────────────────────────
if [[ $ACTION == create ]]; then
  install_miniconda

  if "$CONDA_BIN" env list | grep -qE "^$ENV_NAME\s"; then
    echo "[=] Environment '$ENV_NAME' already exists."
  else
    echo "[+] Creating environment '$ENV_NAME' (PyPy 3.10) ..."
    "$CONDA_BIN" create -y -n "$ENV_NAME" $PKGS
    echo "[✓] Conda packages installed."
  fi

  echo "[+] Installing system Graphviz libs (sudo may prompt) ..."
  sudo apt-get update -qq
  sudo apt-get install -y graphviz graphviz-dev

  echo
  echo "[✓] Done.  Start a new shell or run:"
  echo "       source ~/.bashrc && conda activate $ENV_NAME"
  echo "   Verify with:"
  echo "       python -c 'import platform, pandas; print(platform.python_implementation(), platform.python_version(), pandas.__version__)'"
  exit 0
fi

# ── activate ─────────────────────────────────────────────────────────
if [[ $ACTION == activate ]]; then
  if [[ -x $CONDA_BIN ]] && "$CONDA_BIN" env list | grep -qE "^$ENV_NAME\s"; then
    activate_env
  else
    echo "[!] Environment '$ENV_NAME' does not exist. Run '$0 create' first."
    exit 1
  fi
fi

# ── usage fallback ───────────────────────────────────────────────────
echo "Usage: $0 [create|delete|activate]"
exit 1
