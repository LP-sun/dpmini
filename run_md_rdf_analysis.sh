#!/bin/bash
# All-in-one: MD simulation + RDF analysis workflow

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║         MD Simulation + RDF Analysis Workflow                  ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

MODEL="${1:-exports_cuda_opt/model_cuda_20251228-121805.pth}"
SYSTEM="${2:-collect/data0}"
STEPS="${3:-1000}"
OUTPUT="${4:-analysis}"

echo "Configuration:"
echo "  Model:  $MODEL"
echo "  System: $SYSTEM"
echo "  Steps:  $STEPS"
echo "  Output: $OUTPUT"
echo ""

# Step 1: Run MD simulation
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 1: Running MD simulation..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

conda run -n deepmd python -u md_simulation.py \
    --model "$MODEL" \
    --system "$SYSTEM" \
    --steps "$STEPS" \
    --dt 0.001 \
    --output "${OUTPUT}_md"

if [ $? -ne 0 ]; then
    echo "❌ MD simulation failed!"
    exit 1
fi

echo ""
echo "✓ MD simulation completed"
echo ""

# Step 2: Compute RDF
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 2: Computing RDF..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

conda run -n deepmd python -u compute_rdf.py \
    "${OUTPUT}_md.npz" \
    --pairs O-O O-H H-H \
    --rmax 8.0 \
    --nbins 200 \
    --output "${OUTPUT}_rdf"

if [ $? -ne 0 ]; then
    echo "❌ RDF computation failed!"
    exit 1
fi

echo ""
echo "✓ RDF computation completed"
echo ""

# Step 3: Interpret RDF
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 3: Interpreting RDF..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

conda run -n deepmd python -u interpret_rdf.py "${OUTPUT}_rdf_data.txt"

if [ $? -ne 0 ]; then
    echo "❌ RDF interpretation failed!"
    exit 1
fi

echo ""

# Step 4: Analyze trajectory
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 4: Analyzing trajectory..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

conda run -n deepmd python -u analyze_trajectory.py \
    "${OUTPUT}_md.npz" \
    --save "${OUTPUT}_summary.json"

if [ $? -ne 0 ]; then
    echo "❌ Trajectory analysis failed!"
    exit 1
fi

echo ""

# Summary
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                    ✅ Analysis Complete!                       ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "Generated files:"
echo "  📊 MD Trajectory:   ${OUTPUT}_md.npz"
echo "  📈 RDF Plot:        ${OUTPUT}_rdf_plot.png"
echo "  📝 RDF Data:        ${OUTPUT}_rdf_data.txt"
echo "  📋 MD Summary:      ${OUTPUT}_summary.json"
echo ""
echo "View results:"
echo "  🖼️  RDF plot:        xdg-open ${OUTPUT}_rdf_plot.png"
echo "  📄 RDF data:        cat ${OUTPUT}_rdf_data.txt | head -30"
echo "  📊 Summary:         cat ${OUTPUT}_summary.json"
echo ""
echo "Next steps:"
echo "  - Examine RDF peaks for structural features"
echo "  - Check energy conservation in summary"
echo "  - Run longer simulation for better statistics"
echo ""
