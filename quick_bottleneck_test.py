#!/usr/bin/env python3
"""Quick bottleneck analysis - simplified version."""

import time
import json
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from dpmini import DeepMDDataset


def test_data_loading_speed(num_workers, batch_size=8):
    """Test data loading speed with different num_workers."""
    print(f"\n{'='*80}")
    print(f"Testing with num_workers={num_workers}, batch_size={batch_size}")
    print('='*80)
    
    # Load config
    with open("se_e2_a/input_torch.json", 'r') as f:
        config = json.load(f)
    
    type_map = config['model']['type_map']
    system_dirs = ['collect/data0']
    
    # Create dataset and dataloader
    dataset = DeepMDDataset(system_dirs, type_map=type_map)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if num_workers > 0 else False
    )
    
    # Test pure iteration speed
    print("\nPure data loading (CPU only, no GPU):")
    num_batches = 50
    start = time.time()
    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break
        # Just access the data
        positions, atom_types, box, energy, forces = batch
        _ = positions.shape  # Force actual loading
    elapsed = time.time() - start
    
    steps_per_sec = num_batches / elapsed
    ms_per_batch = (elapsed / num_batches) * 1000
    
    print(f"  Time: {elapsed:.2f}s for {num_batches} batches")
    print(f"  Speed: {steps_per_sec:.2f} steps/s")
    print(f"  Per batch: {ms_per_batch:.1f} ms")
    
    # Test with GPU transfer
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("\nWith GPU transfer:")
        
        start = time.time()
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            positions, atom_types, box, energy, forces = batch
            
            # Transfer to GPU
            positions = positions.to(device, non_blocking=True)
            atom_types = atom_types.to(device, non_blocking=True)
            box = box.to(device, non_blocking=True)
            energy = energy.to(device, non_blocking=True)
            forces = forces.to(device, non_blocking=True)
            torch.cuda.synchronize()
        
        elapsed = time.time() - start
        steps_per_sec = num_batches / elapsed
        ms_per_batch = (elapsed / num_batches) * 1000
        
        print(f"  Time: {elapsed:.2f}s for {num_batches} batches")
        print(f"  Speed: {steps_per_sec:.2f} steps/s")
        print(f"  Per batch: {ms_per_batch:.1f} ms")
    
    return steps_per_sec


def analyze_cpu_usage():
    """Check CPU and memory info."""
    print("\n" + "="*80)
    print("System Resources")
    print("="*80)
    
    import psutil
    print(f"CPU cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical")
    print(f"CPU usage: {psutil.cpu_percent(interval=1)}%")
    print(f"Memory: {psutil.virtual_memory().percent}% used")
    
    if torch.cuda.is_available():
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.memory_allocated(0)/1e9:.2f} GB allocated")


def main():
    print("Performance Bottleneck Analysis")
    print("="*80)
    
    # Check system
    try:
        analyze_cpu_usage()
    except:
        print("(psutil not available, skipping system check)")
    
    # Test different num_workers
    results = {}
    for nw in [0, 2, 4, 8, 12]:
        try:
            speed = test_data_loading_speed(num_workers=nw, batch_size=8)
            results[nw] = speed
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error with num_workers={nw}: {e}")
            continue
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY: Data Loading Speed")
    print("="*80)
    print(f"{'num_workers':<15} {'steps/s':<10}")
    print("-"*25)
    for nw, speed in sorted(results.items()):
        print(f"{nw:<15} {speed:<10.2f}")
    
    if results:
        best_nw = max(results, key=results.get)
        print(f"\nBest: num_workers={best_nw} -> {results[best_nw]:.2f} steps/s")
        print(f"Current training (num_workers=4): ~2.3 steps/s")
        print(f"Potential speedup: {results.get(best_nw, 0) / 2.3:.1f}x")


if __name__ == '__main__':
    main()
