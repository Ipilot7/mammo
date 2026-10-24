#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

START_STAGE=${1:-classifier}
case "$START_STAGE" in
    classifier) START_INDEX=1 ;;
    screening) START_INDEX=2 ;;
    detector) START_INDEX=3 ;;
    evaluation) START_INDEX=4 ;;
    *) echo "usage: $0 [classifier|screening|detector|evaluation]" >&2; exit 2 ;;
esac

export PYTHONUNBUFFERED=1
export TORCH_HOME=/mnt/backup/tools/torch-cache
export HF_HOME=/mnt/backup/tools/hf-cache
export YOLO_CONFIG_DIR=/mnt/backup/tools/ultralytics
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p models/logs
LOG=models/logs/stage7-full-training.log
STATUS=models/logs/stage7-status.txt
if (( START_INDEX == 1 )); then
    : >"$LOG"
    printf 'started=%s\nstate=running\n' "$(date --iso-8601=seconds)" >"$STATUS"
else
    printf '\n===== resume from %s: %s =====\n' "$START_STAGE" "$(date --iso-8601=seconds)" >>"$LOG"
    printf 'resumed=%s\nstate=running\nresume_stage=%s\n' \
        "$(date --iso-8601=seconds)" "$START_STAGE" >>"$STATUS"
fi

finish() {
    rc=$?
    if (( rc == 0 )); then
        state=completed
    else
        state=failed
    fi
    printf 'finished=%s\nstate=%s\nexit_code=%s\n' "$(date --iso-8601=seconds)" "$state" "$rc" >>"$STATUS"
    exit "$rc"
}
trap finish EXIT

run_stage() {
    stage=$1
    shift
    printf '\n===== %s: %s =====\n' "$stage" "$(date --iso-8601=seconds)" | tee -a "$LOG"
    printf 'current_stage=%s\n' "$stage" >>"$STATUS"
    "$@" 2>&1 | tee -a "$LOG"
}

# Train the smaller crop model first, then whole-breast screening, then the long detector run.
if (( START_INDEX <= 1 )); then
    run_stage classifier .venv/bin/python -m train.train_classifier \
        --arch efficientnet_b3 --epochs 30 --batch 16 --workers 4 --device cuda
fi

if (( START_INDEX <= 2 )); then
    run_stage screening .venv/bin/python -m train.train_screening \
        --arch efficientnet_b3 --epochs 12 --samples-per-epoch 6000 \
        --batch 12 --workers 4 --device cuda
fi

if (( START_INDEX <= 3 )); then
    run_stage detector .venv/bin/python -m train.train_detector \
        --model yolo11s.pt --epochs 40 --imgsz 1280 --batch 2 --device 0 --fraction 1.0
fi

run_stage detector_checkpoint .venv/bin/python -c \
    'from train.evaluate import prepare_detector_checkpoint; print(prepare_detector_checkpoint())'

run_stage evaluation .venv/bin/python -m train.evaluate --device cuda

run_stage checksums bash -c '
    cd models
    sha256sum classifier.pt > classifier.pt.sha256
    sha256sum screening.pt > screening.pt.sha256
    sha256sum detector.pt > detector.pt.sha256
    sha256sum -c classifier.pt.sha256 screening.pt.sha256 detector.pt.sha256
'
