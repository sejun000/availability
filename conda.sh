#!/usr/bin/env bash
set -euo pipefail

# ── configuration ────────────────────────────────────────
CONDA_DIR="$HOME/miniconda3"           # 설치·삭제 대상 경로
MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
CONDA_BIN="$CONDA_DIR/bin/conda"
ACTION="${1:-install}"                 # install | delete

# ── functions ────────────────────────────────────────────
install_conda() {
  if [[ ! -d "$CONDA_DIR" ]]; then
    echo "[+] Installing Miniconda into $CONDA_DIR ..."
    wget -q --show-progress "$MINICONDA_URL" -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$CONDA_DIR"
    rm /tmp/miniconda.sh
    echo "[✓] Miniconda installed."
  else
    echo "[=] $CONDA_DIR already exists – skip install."
  fi

  # one-time init & channel 설정 (반복 실행 안전)
  "$CONDA_BIN" init "$(basename "$SHELL")" >/dev/null 2>&1 || true
  "$CONDA_BIN" config --add channels conda-forge 2>/dev/null || true
  "$CONDA_BIN" config --set channel_priority flexible

  echo
  echo "[✓] Miniconda 준비 완료. 새 셸을 열거나:"
  echo "       source ~/.bashrc && conda --version"
}

delete_conda() {
  if [[ -x "$CONDA_BIN" ]]; then
    echo "[+] Removing conda init hooks ..."
    "$CONDA_BIN" init --reverse >/dev/null 2>&1 || true
  fi

  echo "[+] Deleting $CONDA_DIR ..."
  rm -rf "$CONDA_DIR"

  echo "[+] Cleaning user data ~/.conda ~/.cache/conda ..."
  rm -rf ~/.conda ~/.cache/conda ~/.continuum

  # 잔여 initialize 블럭이 남았을 수도 있으니 확인만 안내
  echo
  echo "[i] ~/.bashrc (또는 ~/.zshrc)에 'conda initialize' 블럭이 남았으면 수동 삭제해 주세요."
  echo "[✓] Miniconda 및 관련 파일 삭제 완료."
}

# ── main ────────────────────────────────────────────────
case "$ACTION" in
  install) install_conda ;;
  delete)  delete_conda  ;;
  *)
    echo "Usage: $0 [install|delete]"
    exit 1
    ;;
esac
