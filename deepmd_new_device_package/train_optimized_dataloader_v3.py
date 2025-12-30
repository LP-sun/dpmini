#!/usr/bin/env python3
"""
v3 优化训练脚本：
- 使用 DataLoader（pin_memory=True, prefetch_factor=2, persistent_workers=True）
- 梯度累积（默认4），仅在累积末尾进行 step() 和 zero_grad()
- 保持与原始脚本一致的 step 语义（按样本计步），日志包含 update 计数
"""
import sys
import os
sys.path.insert(0, '/home/ubuntu/pj')
os.chdir('/home/ubuntu/pj')

from train_cuda_optimized import (
    load_config, get_learning_rate, get_loss_prefactors, compute_loss, check_cuda_info
)

import argparse
from pathlib import Path
from datetime import datetime
import torch
from torch.utils.data import DataLoader
from dpmini import DeepMDModel, DeepMDDataset


def main():
    parser = argparse.ArgumentParser(description='Optimized DeepMD training v3 (DataLoader + Accum)')
    parser.add_argument('--config', type=str, default='config_formal_100k_optimized.json')
    parser.add_argument('--data-dir', type=str, default='collect/O64H128')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints_optimized_v3')
    parser.add_argument('--export-dir', type=str, default='exports_optimized_v3')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--mixed-precision', action='store_true')
    parser.add_argument('--grad-accumulation-steps', type=int, default=4)
    parser.add_argument('--force-loss', type=str, default='mse')
    parser.add_argument('--test-steps', type=int, default=0)
    args = parser.parse_args()

    print("Starting optimized DeepMD training v3...", flush=True)
    check_cuda_info()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    config = load_config(config_path)

    model_config = config.get('model', {})
    descriptor_config = model_config.get('descriptor', {})
    fitting_config = model_config.get('fitting_net', {})
    type_map = model_config.get('type_map', ['O', 'H'])

    training_config = config.get('training', {})
    numb_steps = training_config.get('numb_steps', 100000)
    if args.test_steps > 0:
        numb_steps = args.test_steps

    system_dirs = [args.data_dir]
    batch_size = args.batch_size

    print("=" * 80)
    print("TRAINING CONFIGURATION (V3)")
    print("=" * 80)
    print(f"Training data: {system_dirs}")
    print(f"Batch size: {batch_size}")
    print(f"Num workers: {args.num_workers}")
    print(f"Grad accumulation steps: {args.grad_accumulation_steps}")
    print(f"Total steps: {numb_steps}")
    print(f"pin_memory: True, persistent_workers: True, non_blocking: True", flush=True)

    # Dataset & DataLoader
    dataset = DeepMDDataset(system_dirs, type_map=type_map)
    print(f"Dataset: {len(dataset)} frames, {dataset[0][0].shape[0]} atoms per frame")
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=args.num_workers,
        prefetch_factor=2,
        pin_memory=True,
        persistent_workers=True,
        shuffle=True,
        drop_last=False,
    )
    print(f"DataLoader: {len(train_loader)} batches per epoch\n", flush=True)

    # Model
    print("Creating model...", flush=True)
    model = DeepMDModel(
        type_map=type_map,
        rcut=descriptor_config['rcut'],
        rcut_smth=descriptor_config['rcut_smth'],
        sel=descriptor_config['sel'],
        descriptor_neuron=descriptor_config['neuron'],
        axis_neuron=descriptor_config['axis_neuron'],
        fitting_neuron=fitting_config['neuron'],
        type_one_side=descriptor_config.get('type_one_side', True),
        resnet_dt=fitting_config.get('resnet_dt', True),
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n", flush=True)
    model.to(device)

    scaler = torch.cuda.amp.GradScaler() if (args.mixed_precision and torch.cuda.is_available()) else None
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    Path(args.checkpoint_dir).mkdir(exist_ok=True)
    Path(args.export_dir).mkdir(exist_ok=True)

    print("=" * 80)
    print(f"Training for {numb_steps} steps...")
    print("=" * 80, flush=True)

    step = 1  # 按样本计步
    update = 0
    accum_in_bucket = 0
    training_start_time = datetime.now()
    f_rmse_ema = None
    ema_alpha = 0.01

    optimizer.zero_grad(set_to_none=True)

    try:
        epoch = 0
        while step <= numb_steps:
            epoch += 1
            print(f"\n[Epoch {epoch}]", flush=True)
            for batch in train_loader:
                if step > numb_steps:
                    break

                positions_batch, atom_types_batch, boxes_batch, energies_batch, forces_batch = batch
                bs_act = positions_batch.shape[0]

                for b in range(bs_act):
                    if step > numb_steps:
                        break

                    positions = positions_batch[b].to(device, non_blocking=True).requires_grad_(True)
                    atom_types = atom_types_batch[b].to(device, non_blocking=True)
                    box = boxes_batch[b].to(device, non_blocking=True)
                    target_energy = energies_batch[b].to(device, non_blocking=True)
                    target_forces = forces_batch[b].to(device, non_blocking=True)

                    natoms = positions.shape[0]

                    lr = get_learning_rate(step, config, numb_steps)
                    for g in optimizer.param_groups:
                        g['lr'] = lr

                    pref_e, pref_f = get_loss_prefactors(step, numb_steps, config)

                    if scaler is not None:
                        with torch.cuda.amp.autocast(dtype=torch.float16):
                            pred_energy, _, pred_forces = model.get_forces(positions, atom_types, box)
                            loss, e_loss, f_loss, e_rmse, f_rmse = compute_loss(
                                pred_energy, target_energy, pred_forces, target_forces,
                                pref_e, pref_f, natoms, force_loss_type=args.force_loss
                            )
                        loss_scaled = loss / args.grad_accumulation_steps
                        scaler.scale(loss_scaled).backward()
                    else:
                        pred_energy, _, pred_forces = model.get_forces(positions, atom_types, box)
                        loss, e_loss, f_loss, e_rmse, f_rmse = compute_loss(
                            pred_energy, target_energy, pred_forces, target_forces,
                            pref_e, pref_f, natoms, force_loss_type=args.force_loss
                        )
                        loss_scaled = loss / args.grad_accumulation_steps
                        loss_scaled.backward()

                    accum_in_bucket += 1

                    # 仅在累积末尾进行 step/zero_grad
                    if accum_in_bucket % args.grad_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        if scaler is not None:
                            scaler.unscale_(optimizer)
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        update += 1

                    # 更新EMA与日志
                    f_rmse_val = f_rmse.item()
                    f_rmse_ema = f_rmse_val if f_rmse_ema is None else (1 - ema_alpha) * f_rmse_ema + ema_alpha * f_rmse_val

                    if step % 100 == 0 or step <= 5:
                        elapsed = (datetime.now() - training_start_time).total_seconds()
                        speed = step / elapsed if elapsed > 0 else 0.0
                        remain = max(numb_steps - step, 0)
                        eta_h = (remain / speed / 3600) if speed > 0 else 0.0
                        print(
                            f"Step {step:6d} (update={update:6d}) | lr={lr:.3e} | loss={loss.item():.4f} | "
                            f"e_rmse={e_rmse.item():.4f} f_rmse={f_rmse.item():.4f} f_ema={f_rmse_ema:.4f} | "
                            f"{speed:.2f} step/s | ETA: {eta_h:.1f}h",
                            flush=True,
                        )

                    # 保存检查点（按样本步数对齐）
                    if step % 10000 == 0:
                        ckpt_path = Path(args.checkpoint_dir) / f'model_step{step}.pt'
                        torch.save(model.state_dict(), ckpt_path)
                        print(f"  Checkpoint saved: {ckpt_path}", flush=True)

                    step += 1

                # end for sample
            # end for batch

        # 训练循环结束
        # 若末尾有未完成的累积，可选择性补一次step（通常影响极小，这里忽略）

    except KeyboardInterrupt:
        print("\nInterrupted by user", flush=True)
    except Exception as e:
        print(f"\nError: {e}", flush=True)
        import traceback
        traceback.print_exc()

    # 导出
    print(f"\nTraining completed at step {step}", flush=True)
    print(f"Total time: {(datetime.now() - training_start_time).total_seconds() / 3600:.2f}h", flush=True)
    export_path = Path(args.export_dir) / 'model_final.pth'
    torch.save(model.state_dict(), export_path)
    print(f"Model exported to {export_path}")
    print(f"Checkpoints: {args.checkpoint_dir}")
    print(f"Exports: {args.export_dir}")


if __name__ == '__main__':
    main()
