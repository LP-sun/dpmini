#!/usr/bin/env python3
"""
批量评估多个 checkpoint
评估最新的 5 个 checkpoint 并保存结果到 CSV
"""
import subprocess
import json
import csv
from pathlib import Path
import re

def extract_step_from_checkpoint(ckpt_file):
    """从 checkpoint 文件名提取 step 数字"""
    match = re.search(r'model_step(\d+)', ckpt_file)
    if match:
        return int(match.group(1))
    return None

def evaluate_all_checkpoints(ckpt_dir, output_csv, config_path='config_formal_100k_fixed.json',
                            system_dir='/home/ubuntu/pj/collect/O64H128',
                            val_indices='/home/ubuntu/pj/logs/val_indices.npy',
                            n_latest=5):
    """
    评估最新的 n 个 checkpoint
    
    Args:
        ckpt_dir: checkpoint 目录
        output_csv: 输出 CSV 文件
        config_path: 配置文件路径
        system_dir: 数据系统目录
        val_indices: 验证集索引文件
        n_latest: 评估最新的几个 checkpoint
    """
    
    ckpt_dir = Path(ckpt_dir)
    
    # 获取所有 checkpoint 文件
    ckpt_files = sorted(ckpt_dir.glob('model_step*.pt'))
    
    if not ckpt_files:
        print(f"❌ 在 {ckpt_dir} 中未找到 checkpoint 文件")
        return
    
    # 按 step 数排序并取最新的 n 个
    ckpts_with_steps = []
    for ckpt_file in ckpt_files:
        step = extract_step_from_checkpoint(str(ckpt_file))
        if step is not None:
            ckpts_with_steps.append((step, ckpt_file))
    
    ckpts_with_steps.sort(key=lambda x: x[0], reverse=True)
    latest_ckpts = ckpts_with_steps[:n_latest]
    
    print(f"\n📊 批量评估 {ckpt_dir.name} 的 {len(latest_ckpts)} 个最新 checkpoint")
    print("=" * 70)
    
    results = []
    
    for step, ckpt_file in sorted(latest_ckpts, key=lambda x: x[0]):
        print(f"\n📝 评估: {ckpt_file.name} (Step {step})")
        print("-" * 70)
        
        cmd = [
            'python', 'evaluate_checkpoint_fast.py',
            '--checkpoint', str(ckpt_file),
            '--system', system_dir,
            '--val-indices', val_indices,
            '--config', config_path,
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd='/home/ubuntu/pj')
            
            if result.returncode == 0:
                # 提取 JSON 结果（最后一行）
                output_lines = result.stdout.strip().split('\n')
                json_line = output_lines[-1]
                
                try:
                    metrics = json.loads(json_line)
                    metrics['step'] = step
                    results.append(metrics)
                    
                    print(f"✓ E_RMSE: {metrics['e_rmse_per_atom']:.6f}")
                    print(f"✓ F_RMSE: {metrics['f_rmse']:.6f}")
                    print(f"✓ F_RMSE_tail: {metrics['f_rmse_tail']:.6f}")
                except json.JSONDecodeError as e:
                    print(f"❌ 解析 JSON 失败: {e}")
                    print(f"Last line: {json_line}")
            else:
                print(f"❌ 评估失败，返回码: {result.returncode}")
                if result.stderr:
                    print(f"错误信息: {result.stderr[:300]}")
        except subprocess.TimeoutExpired:
            print(f"❌ 评估超时 (>600s)")
        except Exception as e:
            print(f"❌ 评估异常: {e}")
    
    # 保存结果到 CSV
    if results:
        results.sort(key=lambda x: x['step'])
        
        with open(output_csv, 'w', newline='') as f:
            fieldnames = ['step', 'e_rmse_per_atom', 'f_rmse', 'f_rmse_tail', 'n_val_frames']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        
        print(f"\n✓ 结果保存到: {output_csv}")
        print("\n📊 评估结果汇总:")
        print("=" * 70)
        print(f"{'Step':<10} {'E_RMSE':<15} {'F_RMSE':<15} {'F_RMSE_tail':<15}")
        print("-" * 70)
        for r in results:
            print(f"{r['step']:<10} {r['e_rmse_per_atom']:<15.6f} {r['f_rmse']:<15.6f} {r['f_rmse_tail']:<15.6f}")
        print("=" * 70)
    else:
        print(f"❌ 没有成功评估任何 checkpoint")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Batch evaluate checkpoints')
    parser.add_argument('--ckpt-dir', type=str, default='checkpoints_fixed', help='Checkpoint directory')
    parser.add_argument('--output', type=str, default='logs/ckpt_eval_fixed.csv', help='Output CSV file')
    parser.add_argument('--config', type=str, default='config_formal_100k_fixed.json', help='Config file')
    parser.add_argument('--n-latest', type=int, default=5, help='Number of latest checkpoints to evaluate')
    parser.add_argument('--system', type=str, default='/home/ubuntu/pj/collect/O64H128', help='System directory')
    parser.add_argument('--val-indices', type=str, default='/home/ubuntu/pj/logs/val_indices.npy', help='Validation indices file')
    
    args = parser.parse_args()
    
    evaluate_all_checkpoints(
        args.ckpt_dir,
        args.output,
        config_path=args.config,
        system_dir=args.system,
        val_indices=args.val_indices,
        n_latest=args.n_latest
    )
