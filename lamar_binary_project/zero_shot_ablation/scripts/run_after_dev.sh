#!/usr/bin/env bash
set -euo pipefail

RUN=${LAMAR_WORK_ROOT}/lamar_zero_shot_ablation/run_20260723T045155Z
PY=${LAMAR_ENV}/bin/python
CONFIG="$RUN/configs/config.json"
export PYTHONPATH="$RUN/scripts:${LAMAR_SOURCE_ROOT}"

test -f "$RUN/DEV_SELECTION_COMPLETE"
test ! -e "$RUN/CALIBRATION_COMPLETE"
test ! -e "$RUN/TEST_EVALUATION_STARTED"

echo "$(date -u +%FT%TZ) extracting calibration embeddings"
CUDA_VISIBLE_DEVICES=0 "$PY" "$RUN/scripts/extract_embeddings.py" \
    --config "$CONFIG" --split calibration \
    --output "$RUN/embeddings/calibration.parquet" \
    > "$RUN/logs/extract_calibration.log" 2>&1

echo "$(date -u +%FT%TZ) evaluating frozen reference models on calibration"
CUDA_VISIBLE_DEVICES=0 "$PY" "$RUN/scripts/predict_reference_models.py" \
    --config "$CONFIG" --split calibration \
    --embedding "$RUN/embeddings/calibration.parquet" \
    --output "$RUN/predictions/reference_calibration.parquet" \
    > "$RUN/logs/reference_calibration.log" 2>&1

echo "$(date -u +%FT%TZ) fitting calibration models and freezing thresholds"
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 "$PY" "$RUN/scripts/calibrate_and_evaluate.py" \
    --config "$CONFIG" --stage calibration \
    > "$RUN/logs/calibration.log" 2>&1
test -f "$RUN/CALIBRATION_COMPLETE"
test -f "$RUN/PRETEST_FROZEN.json"

echo "$(date -u +%FT%TZ) beginning one pre-registered locked-test suite"
printf '%s\n' "$(date -u +%FT%TZ) configurations/calibrators/thresholds frozen before test" \
    > "$RUN/TEST_EVALUATION_STARTED"

CUDA_VISIBLE_DEVICES=0 "$PY" "$RUN/scripts/extract_embeddings.py" \
    --config "$CONFIG" --split test \
    --output "$RUN/embeddings/test.parquet" \
    > "$RUN/logs/extract_test.log" 2>&1

CUDA_VISIBLE_DEVICES=0 "$PY" "$RUN/scripts/predict_reference_models.py" \
    --config "$CONFIG" --split test \
    --embedding "$RUN/embeddings/test.parquet" \
    --output "$RUN/predictions/reference_test.parquet" \
    > "$RUN/logs/reference_test.log" 2>&1

"$PY" "$RUN/scripts/calibrate_and_evaluate.py" \
    --config "$CONFIG" --stage test \
    > "$RUN/logs/test_evaluation.log" 2>&1
test -f "$RUN/TEST_EVALUATION_COMPLETE"

echo "$(date -u +%FT%TZ) generating figures, report, checksums, and SUCCESS"
"$PY" "$RUN/scripts/finalize.py" --config "$CONFIG" \
    > "$RUN/logs/finalize.log" 2>&1
test -f "$RUN/SUCCESS"
echo "$(date -u +%FT%TZ) pipeline complete"
