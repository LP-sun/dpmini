#!/bin/bash
cd /home/ubuntu/pj

echo "Training Status Check"
echo "===================="
echo ""

# Check process
echo "Active Training Processes:"
pgrep -a python | grep train_deepmd || echo "  (none found)"

echo ""
echo "Recent Log Files:"
ls -lh *.log | tail -n 5

echo ""
echo "Quick Test Log Size:"
stat -c%s quick_test.log 2>/dev/null || echo "  file not created yet"

echo ""
echo "Last 100 lines of quick_test.log:"
tail -n 100 quick_test.log 2>/dev/null || echo "  (file empty or not found)"

echo ""
echo "GPU Status:"
nvidia-smi -i 0 --query-gpu=utilization.gpu,memory.used --format=csv,noheader || echo "  (error checking GPU)"

echo ""
echo "To monitor training in real-time:"
echo "  tail -f quick_test.log"
