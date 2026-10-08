#!/usr/bin/env bash
# 한 명령으로 이 PC에 맞는 PyTorch 환경 설치 + 확인 (Ubuntu/Linux).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-auto}"
case "$MODE" in
  auto)
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
      MODE=cuda
    else
      MODE=cpu
    fi
    ;;
  cpu|cuda) ;;
  *) echo "사용법: bash scripts/setup_env.sh [auto|cpu|cuda]" >&2; exit 2 ;;
esac

# 컴퓨터별 환경을 공유하지 않으며 CPU/CUDA도 서로 독립된 폴더로 관리한다.
VENV="$ROOT/.venv-$MODE"
echo "선택: $MODE | 가상환경: $VENV"
if [[ ! -x "$VENV/bin/python" ]]; then
  python3 -m venv "$VENV" || {
    echo "venv 생성 실패: Ubuntu라면 sudo apt install python3-venv를 확인하세요." >&2
    exit 1
  }
fi
PY="$VENV/bin/python"
"$PY" -m pip install --upgrade pip
if [[ "$MODE" == cpu ]]; then
  TORCH_INDEX="https://download.pytorch.org/whl/cpu"
else
  # CUDA 빌드 선택: NVIDIA driver는 호스트에 설치되어 있어야 한다.
  TORCH_INDEX="https://download.pytorch.org/whl/cu128"
fi
"$PY" -m pip install torch torchvision --index-url "$TORCH_INDEX"
"$PY" -m pip install -r "$ROOT/requirements.txt"

"$PY" - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("PyTorch CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
if [[ "$MODE" == cuda ]]; then
  "$PY" -c 'import torch; assert torch.cuda.is_available(), "GPU가 감지되었으나 CUDA를 사용할 수 없습니다. NVIDIA 드라이버와 PyTorch 호환성을 점검하세요."'
fi
echo
echo "설치 완료. 실행 예시:"
echo "  $VENV/bin/python $ROOT/study/05_vit.py"
echo "활성화: source $VENV/bin/activate"
