#!/bin/bash
# Run short MD simulation with converted LAMMPS structure

set -euo pipefail

# Configuration
MODEL="exports_final/model_cuda_20251228-171614.pth"
SYSTEM="collect/data_lammps"
STEPS=5000
DT=0.001
OUTPUT="short_md_from_lammps"
CHECKPOINT_INTERVAL=1000
LOG_FILE="${OUTPUT}.log"

echo "=========================================="
echo "MD Simulation from Converted LAMMPS File"
echo "=========================================="
echo "Model: $MODEL"
echo "System: $SYSTEM"
echo "Steps: $STEPS"
echo "Time step: $DT ps"
echo "Total time: $(echo "$STEPS * $DT" | bc) ps"
echo "Output prefix: $OUTPUT"
echo "Log file: $LOG_FILE"
echo ""

# Run MD simulation with unbuffered output (stdout+stderr to both console and file)
# Use stdbuf to force line-buffering across the pipeline
set +e
stdbuf -oL -eL conda run -n deepmd python3 -u md_simulation.py \
        --model "$MODEL" \
        --system "$SYSTEM" \
        --steps $STEPS \
        --dt $DT \
        --output "$OUTPUT" \
        --checkpoint-interval $CHECKPOINT_INTERVAL \
        --device cuda 2>&1 | tee "$LOG_FILE"
rc=${PIPESTATUS[0]}
set -e

echo ""
echo "=========================================="
if [[ $rc -eq 0 ]]; then
    echo "✓ MD simulation completed!"
else
    echo "✗ MD simulation failed with exit code $rc"
fi
echo "=========================================="
echo ""
echo "Output files:"
ls -lh ${OUTPUT}.npz 2>/dev/null || true
ls -lh ${OUTPUT}_checkpoint_*.npz 2>/dev/null | head -5 || true

