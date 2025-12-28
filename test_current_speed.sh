#!/bin/bash
# Monitor current training speed in real-time

while true; do
    clear
    echo "=== Training Monitor ==="
    echo "Time: $(date)"
    echo ""
    
    if [ -f "training_optimized.log" ]; then
        echo "Recent Steps:"
        grep "Step" training_optimized.log | tail -5 | sed 's/^/  /'
        echo ""
        
        # Extract latest speed and GPU util
        LATEST=$(grep "Step" training_optimized.log | tail -1)
        SPEED=$(echo "$LATEST" | grep -oP 'speed \K[0-9.]+' | head -1)
        GPU=$(echo "$LATEST" | grep -oP 'util \K[0-9]+')
        STEPS=$(echo "$LATEST" | grep -oP 'Step\s+\K[0-9]+' | head -1)
        
        if [ -n "$SPEED" ]; then
            echo "Current Speed: $SPEED steps/s"
            echo "GPU Utilization: ${GPU}%"
            
            # Calculate ETA (assuming 100000 total steps)
            REMAINING=$((100000 - ${STEPS:-0}))
            ETA_SEC=$(echo "scale=0; $REMAINING / $SPEED" | bc 2>/dev/null || echo "?")
            ETA_HOUR=$(echo "scale=1; $ETA_SEC / 3600" | bc 2>/dev/null || echo "?")
            echo "ETA: ~${ETA_HOUR}h (or $(($ETA_SEC / 60)) min)"
        fi
    fi
    
    echo ""
    echo "GPU Status:"
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader | sed 's/^/  /'
    
    echo ""
    echo "(Refreshing every 10s, press Ctrl+C to exit)"
    sleep 10
done
