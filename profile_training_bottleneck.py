#!/usr/bin/env python3
"""Profile training to identify performance bottlenecks."""

import time
import json
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from dpmini import DeepMDModel, DeepMDDataset


def profile_data_loading(dataloader, num_batches=10):
    """Profile data loading time."""
    print("=" * 80)
    print("Profiling Data Loading")
    print("=" * 80)
    
    times = []
    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break
        
        start = time.time()
        # Just iterate, don't move to GPU yet
        positions, atom_types, box, energy, forces = batch
        end = time.time()
        times.append(end - start)
        print(f"Batch {i+1}: {(end-start)*1000:.1f} ms")
    
    avg_time = np.mean(times) * 1000
    print(f"\nAverage data loading time: {avg_time:.1f} ms/batch")
    return avg_time


def profile_gpu_transfer(dataloader, device, num_batches=10):
    """Profile GPU data transfer time."""
    print("\n" + "=" * 80)
    print("Profiling GPU Transfer")
    print("=" * 80)
    
    times = []
    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break
        
        positions, atom_types, box, energy, forces = batch
        
        start = time.time()
        positions = positions.to(device)
        atom_types = atom_types.to(device)
        box = box.to(device)
        energy = energy.to(device)
        forces = forces.to(device)
        torch.cuda.synchronize()
        end = time.time()
        
        times.append(end - start)
        print(f"Batch {i+1}: {(end-start)*1000:.1f} ms")
    
    avg_time = np.mean(times) * 1000
    print(f"\nAverage GPU transfer time: {avg_time:.1f} ms/batch")
    return avg_time


def profile_forward_pass(model, dataloader, device, num_batches=10):
    """Profile forward pass time."""
    print("\n" + "=" * 80)
    print("Profiling Forward Pass")
    print("=" * 80)
    
    model.eval()
    times = []
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            
            positions, atom_types, box, energy, forces = batch
            positions = positions.to(device)
            atom_types = atom_types.to(device)
            box = box.to(device)
            
            start = time.time()
            pred_energy, pred_forces = model(positions, atom_types, box)
            torch.cuda.synchronize()
            end = time.time()
            
            times.append(end - start)
            print(f"Batch {i+1}: {(end-start)*1000:.1f} ms")
    
    avg_time = np.mean(times) * 1000
    print(f"\nAverage forward pass time: {avg_time:.1f} ms/batch")
    return avg_time


def profile_backward_pass(model, dataloader, device, num_batches=10):
    """Profile backward pass time."""
    print("\n" + "=" * 80)
    print("Profiling Backward Pass (Forward + Backward + Optimizer)")
    print("=" * 80)
    
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    times = []
    forward_times = []
    backward_times = []
    optimizer_times = []
    
    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break
        
        positions, atom_types, box, energy, forces = batch
        positions = positions.to(device)
        atom_types = atom_types.to(device)
        box = box.to(device)
        energy = energy.to(device)
        forces = forces.to(device)
        
        # Forward
        start = time.time()
        pred_energy, pred_forces = model(positions, atom_types, box)
        loss = torch.nn.functional.mse_loss(pred_energy, energy) + \
               torch.nn.functional.mse_loss(pred_forces, forces)
        torch.cuda.synchronize()
        forward_time = time.time() - start
        forward_times.append(forward_time)
        
        # Backward
        start = time.time()
        optimizer.zero_grad()
        loss.backward()
        torch.cuda.synchronize()
        backward_time = time.time() - start
        backward_times.append(backward_time)
        
        # Optimizer step
        start = time.time()
        optimizer.step()
        torch.cuda.synchronize()
        optimizer_time = time.time() - start
        optimizer_times.append(optimizer_time)
        
        total_time = forward_time + backward_time + optimizer_time
        times.append(total_time)
        
        print(f"Batch {i+1}: Total {total_time*1000:.1f} ms "
              f"(fwd {forward_time*1000:.1f} + bwd {backward_time*1000:.1f} + opt {optimizer_time*1000:.1f})")
    
    avg_forward = np.mean(forward_times) * 1000
    avg_backward = np.mean(backward_times) * 1000
    avg_optimizer = np.mean(optimizer_times) * 1000
    avg_total = np.mean(times) * 1000
    
    print(f"\nAverage times:")
    print(f"  Forward:   {avg_forward:.1f} ms/batch")
    print(f"  Backward:  {avg_backward:.1f} ms/batch")
    print(f"  Optimizer: {avg_optimizer:.1f} ms/batch")
    print(f"  Total:     {avg_total:.1f} ms/batch")
    
    return avg_total, avg_forward, avg_backward, avg_optimizer


def main():
    # Load config
    config_path = Path("se_e2_a/input_torch.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Extract settings
    model_config = config['model']
    training_config = config['training']
    type_map = model_config['type_map']
    descriptor_config = model_config['descriptor']
    fitting_config = model_config['fitting_net']
    
    system_dirs = [str(Path("collect") / Path(training_config['training_data']['systems'][0]).name)]
    batch_size = training_config['training_data']['batch_size']
    
    print("Configuration:")
    print(f"  Systems: {system_dirs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Type map: {type_map}")
    
    # Create dataset with different num_workers
    for num_workers in [0, 2, 4, 8]:
        print("\n" + "=" * 80)
        print(f"Testing with num_workers={num_workers}")
        print("=" * 80)
        
        dataset = DeepMDDataset(system_dirs, type_map=type_map)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False,
            persistent_workers=True if num_workers > 0 else False
        )
        
        # Profile data loading
        data_load_time = profile_data_loading(dataloader, num_batches=10)
        
        # Create model for GPU tests
        if torch.cuda.is_available():
            device = torch.device('cuda')
            model = DeepMDModel(
                type_map=type_map,
                rcut=descriptor_config['rcut'],
                rcut_smth=descriptor_config['rcut_smth'],
                sel=descriptor_config['sel'],
                descriptor_neuron=descriptor_config['neuron'],
                axis_neuron=descriptor_config['axis_neuron'],
                fitting_neuron=fitting_config['neuron'],
                type_one_side=descriptor_config.get('type_one_side', True),
                resnet_dt=fitting_config.get('resnet_dt', True)
            ).to(device)
            
            # Profile GPU transfer
            transfer_time = profile_gpu_transfer(dataloader, device, num_batches=10)
            
            # Profile forward pass
            forward_time = profile_forward_pass(model, dataloader, device, num_batches=10)
            
            # Profile full training step
            total_time, fwd, bwd, opt = profile_backward_pass(model, dataloader, device, num_batches=10)
            
            print("\n" + "=" * 80)
            print(f"SUMMARY for num_workers={num_workers}")
            print("=" * 80)
            print(f"Data Loading:        {data_load_time:.1f} ms/batch")
            print(f"GPU Transfer:        {transfer_time:.1f} ms/batch")
            print(f"Forward Pass:        {forward_time:.1f} ms/batch")
            print(f"Training Step Total: {total_time:.1f} ms/batch")
            print(f"  - Forward:         {fwd:.1f} ms")
            print(f"  - Backward:        {bwd:.1f} ms")
            print(f"  - Optimizer:       {opt:.1f} ms")
            
            total_per_step = data_load_time + transfer_time + total_time
            steps_per_sec = 1000.0 / total_per_step
            print(f"\nEstimated total:     {total_per_step:.1f} ms/step")
            print(f"Estimated speed:     {steps_per_sec:.2f} steps/s")
            
            # Breakdown percentages
            print(f"\nTime breakdown:")
            print(f"  Data Loading: {data_load_time/total_per_step*100:.1f}%")
            print(f"  GPU Transfer: {transfer_time/total_per_step*100:.1f}%")
            print(f"  Computation:  {total_time/total_per_step*100:.1f}%")


if __name__ == '__main__':
    main()
