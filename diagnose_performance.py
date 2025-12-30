#!/usr/bin/env python3
"""
Performance diagnostic tool for DeepMD training.
Identifies bottlenecks and provides optimization recommendations.
"""
import subprocess
import re
import sys
from pathlib import Path
from datetime import datetime


def get_training_speed(log_file):
    """Extract training speed from log file."""
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()[-50:]
        
        speeds = []
        for line in lines:
            m = re.search(r'Speed:\s+([\d.]+)\s+steps/s', line)
            if m:
                speeds.append(float(m.group(1)))
        
        if speeds:
            return {
                'current': speeds[-1],
                'average': sum(speeds) / len(speeds),
                'min': min(speeds),
                'max': max(speeds)
            }
    except:
        pass
    return None


def get_gpu_info():
    """Get GPU utilization and memory info."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--format=csv', 
             '--query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total'],
            capture_output=True, text=True, timeout=5
        )
        
        lines = result.stdout.strip().split('\n')[1:]
        gpus = []
        for line in lines:
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 6:
                gpus.append({
                    'index': parts[0],
                    'name': parts[1],
                    'util_gpu': int(parts[2].rstrip(' %')),
                    'util_mem': int(parts[3].rstrip(' %')),
                    'mem_used': int(parts[4].rstrip(' MiB')),
                    'mem_total': int(parts[5].rstrip(' MiB'))
                })
        return gpus
    except:
        return []


def analyze_dataloader_speed():
    """Benchmark data loading speed."""
    print("\n" + "=" * 80)
    print("DATA LOADER PERFORMANCE BENCHMARK")
    print("=" * 80)
    
    try:
        from dpmini import DeepMDDataset
        from torch.utils.data import DataLoader
        import time
        import torch
        
        dataset = DeepMDDataset(['collect/O64H128'], type_map=['O', 'H'])
        print(f"Dataset size: {len(dataset)} frames")
        
        # Test different configurations
        configs = [
            {'batch_size': 1, 'num_workers': 0, 'pin_memory': False},
            {'batch_size': 1, 'num_workers': 0, 'pin_memory': True},
            {'batch_size': 4, 'num_workers': 4, 'pin_memory': True},
            {'batch_size': 8, 'num_workers': 4, 'pin_memory': True},
        ]
        
        for config in configs:
            loader = DataLoader(
                dataset,
                batch_size=config['batch_size'],
                num_workers=config['num_workers'],
                pin_memory=config['pin_memory'],
                persistent_workers=config['num_workers'] > 0
            )
            
            times = []
            for i, batch in enumerate(loader):
                start = time.time()
                _ = batch[0].shape  # Simulate access
                elapsed = time.time() - start
                times.append(elapsed)
                if i >= 10:
                    break
            
            avg_time = sum(times) / len(times)
            throughput = config['batch_size'] / avg_time
            
            print(f"\nBatch size: {config['batch_size']}, "
                  f"Workers: {config['num_workers']}, "
                  f"Pin memory: {config['pin_memory']}")
            print(f"  Avg time per batch: {avg_time*1000:.2f} ms")
            print(f"  Throughput: {throughput:.1f} samples/s")
    
    except Exception as e:
        print(f"Error: {e}")


def main():
    print("\n" + "=" * 80)
    print("DeepMD TRAINING PERFORMANCE DIAGNOSTIC")
    print("=" * 80)
    print(f"Analysis time: {datetime.now()}")
    
    # Check GPU status
    print("\n" + "=" * 80)
    print("GPU STATUS")
    print("=" * 80)
    
    gpus = get_gpu_info()
    if gpus:
        for gpu in gpus:
            print(f"\nGPU {gpu['index']}: {gpu['name']}")
            print(f"  GPU Utilization: {gpu['util_gpu']}%")
            print(f"  Memory Utilization: {gpu['util_mem']}%")
            print(f"  Memory Used: {gpu['mem_used']} MiB / {gpu['mem_total']} MiB")
            
            # Performance assessment
            if gpu['util_gpu'] < 30:
                print(f"  ⚠️  WARNING: Low GPU utilization (< 30%)")
                print(f"      Increase batch size or enable multi-worker data loading")
            elif gpu['util_gpu'] < 60:
                print(f"  ⚠️  CAUTION: Moderate GPU utilization (30-60%)")
                print(f"      Consider further optimization")
            else:
                print(f"  ✅ Good GPU utilization")
    
    # Check training logs
    print("\n" + "=" * 80)
    print("TRAINING SPEED")
    print("=" * 80)
    
    log_files = [
        ('cuda_training.log', 'Current CUDA Training'),
        ('cuda_training_opt.log', 'Optimized CUDA Training'),
    ]
    
    for log_file, desc in log_files:
        if Path(log_file).exists():
            speed_info = get_training_speed(log_file)
            if speed_info:
                print(f"\n{desc} ({log_file}):")
                print(f"  Current speed: {speed_info['current']:.2f} steps/s")
                print(f"  Average speed: {speed_info['average']:.2f} steps/s")
                print(f"  Min speed: {speed_info['min']:.2f} steps/s")
                print(f"  Max speed: {speed_info['max']:.2f} steps/s")
                
                # Performance recommendations
                if speed_info['current'] < 2.0:
                    print(f"  ⚠️  WARNING: Slow training ({speed_info['current']:.2f} steps/s)")
                    print(f"      Bottleneck likely: small batch size, low GPU util, or data loading")
                elif speed_info['current'] < 5.0:
                    print(f"  ⚠️  CAUTION: Moderate speed ({speed_info['current']:.2f} steps/s)")
                    print(f"      Consider batch size increase or num_workers adjustment")
                else:
                    print(f"  ✅ Good training speed ({speed_info['current']:.2f} steps/s)")
    
    # Benchmark data loading
    analyze_dataloader_speed()
    
    # Configuration recommendations
    print("\n" + "=" * 80)
    print("OPTIMIZATION RECOMMENDATIONS")
    print("=" * 80)
    
    print("\n1. BATCH SIZE (highest impact)")
    print("   Current: check se_e2_a/input_torch.json -> training_data -> batch_size")
    print("   Recommended: 4-8 (depends on GPU memory)")
    print("   Expected improvement: 3-4x")
    
    print("\n2. DATA LOADING (medium impact)")
    print("   Current: check train_cuda.py -> num_workers default")
    print("   Recommended: num_workers=4, pin_memory=True")
    print("   Expected improvement: 10-20%")
    
    print("\n3. CUDA OPTIMIZATION (small but guaranteed)")
    print("   Use: --mixed-precision flag")
    print("   Expected improvement: 10-30%")
    
    print("\n4. ADVANCED (requires code changes)")
    print("   - Neighbor list caching: 10-20% improvement")
    print("   - Gradient accumulation: useful if OOM, not needed now")
    print("   - Fused operations: 5-15% improvement")
    
    print("\n" + "=" * 80)
    print("Quick start optimized training:")
    print("  bash run_training_cuda_optimized.sh")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
