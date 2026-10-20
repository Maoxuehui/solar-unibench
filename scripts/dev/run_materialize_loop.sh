#!/usr/bin/env bash
# Auto-restart loop for Folsom sky image materialization.
# Restarts the python process after each env kill (>20 min) until the zarr
# has both `images` and `images_timestamps` arrays (= fully complete).
#
# Usage: bash scripts/run_materialize_loop.sh [YEAR]
# Default YEAR=2014.
set -u
YEAR="${1:-2014}"
PY="/d/anaconda_data/envs/solar_uni/python.exe"
export TMPDIR=D:/tmp TEMP=D:/tmp TMP=D:/tmp PYTHONUNBUFFERED=1
cd /d/monica/workspace/solar-unibench

OUT="data/standardized/folsom/sky_images_${YEAR}.zarr"
LOG="logs/folsom_${YEAR}_loop.log"
ATTEMPT=0

echo "[$(date)] materialize loop START: year=${YEAR} -> ${OUT}" | tee -a "$LOG"

while true; do
    ATTEMPT=$((ATTEMPT + 1))

    # Check completion: zarr exists + has both arrays.
    if [ -d "${OUT}/.zgroup" ] && [ -d "${OUT}/images_timestamps" ]; then
        N=$("$PY" -c "import zarr; z=zarr.open('${OUT}',mode='r'); print(z['images'].shape[0])" 2>/dev/null || echo "?")
        echo "[$(date)] COMPLETED: ${N} frames in ${OUT} after ${ATTEMPT} attempts" | tee -a "$LOG"
        break
    fi

    echo "[$(date)] attempt ${ATTEMPT}: launching python..." | tee -a "$LOG"
    "$PY" -X faulthandler scripts/rebuild_sky_zarr.py \
        --dataset folsom --only-year "${YEAR}" --limit 0 >> "$LOG" 2>&1
    EXIT_CODE=$?
    echo "[$(date)] attempt ${ATTEMPT}: exited code=${EXIT_CODE}" | tee -a "$LOG"

    if [ $EXIT_CODE -eq 0 ]; then
        echo "[$(date)] python exited cleanly (code 0) — checking completion..." | tee -a "$LOG"
        # The build function returns 0 on completion. Verify.
        if [ -d "${OUT}/images_timestamps" ]; then
            echo "[$(date)] COMPLETED (timestamps array present)" | tee -a "$LOG"
            break
        fi
    fi

    echo "[$(date)] restarting in 3s..." | tee -a "$LOG"
    sleep 3
done
echo "[$(date)] materialize loop DONE" | tee -a "$LOG"
