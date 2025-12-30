#!/bin/bash
echo "Testing model_final.pth on all datasets in collect/"
echo "=================================================="
{
for ds in collect/O64H128 collect/O128H256 collect/data0 collect/data2 collect/water; do
  echo ""
  echo "▶ Dataset: $ds"
  conda run -n ai4m python test_model_on_data.py --model exports_cuda/model_final.pth --data-dir "$ds" 2>&1 | tail -20
done
} | tee test_all_results.txt
echo "Done. Results saved to test_all_results.txt"
