#!/usr/bin/env bash
set -euo pipefail

RUN=${LAMAR_WORK_ROOT}/lamar_binary_models/run_20260722T203752Z
PY=${LAMAR_ENV}/bin/python
MASTER="$RUN/configs/master.yaml"
ORCHESTRATOR_PID="$(cat "$RUN/logs/orchestrate.pid")"
export PYTHONPATH="$RUN/scripts:${LAMAR_SOURCE_ROOT}"

echo "$(date -u +%FT%TZ) waiting for dev orchestrator PID $ORCHESTRATOR_PID"
while ps -p "$ORCHESTRATOR_PID" >/dev/null 2>&1; do
    sleep 60
done

echo "$(date -u +%FT%TZ) dev orchestrator exited"
test -f "$RUN/DEV_SELECTION_COMPLETE"
test -f "$RUN/BEST_DEV_CONFIG.json"

echo "$(date -u +%FT%TZ) starting calibration-only stage"
CUDA_VISIBLE_DEVICES=0 "$PY" "$RUN/scripts/calibrate_and_test.py" --master "$MASTER" --stage calibration
test -f "$RUN/CALIBRATION_COMPLETE"
test -f "$RUN/final_threshold.json"

echo "$(date -u +%FT%TZ) starting the single locked-test evaluation"
CUDA_VISIBLE_DEVICES=0 "$PY" "$RUN/scripts/calibrate_and_test.py" --master "$MASTER" --stage test
test -f "$RUN/TEST_EVALUATION_COMPLETE"

echo "$(date -u +%FT%TZ) finalizing report and checksums"
echo "$(date -u +%FT%TZ) complete after successful finalization"
"$PY" "$RUN/scripts/finalize_report.py" --master "$MASTER"
test -f "$RUN/SUCCESS"
