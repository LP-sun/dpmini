#!/usr/bin/env python3
"""
验证集评估决策逻辑
基于验证集结果判断是否应该继续训练
"""
import csv
from pathlib import Path
import numpy as np

def check_convergence(csv_file, metric='f_rmse', improvement_threshold=0.01, window_size=3):
    """
    检查训练是否已收敛
    
    Args:
        csv_file: 评估结果 CSV 文件
        metric: 检查的指标（e_rmse_per_atom, f_rmse, f_rmse_tail）
        improvement_threshold: 改进阈值（小数），例如 0.01 表示 1%
        window_size: 检查最后 N 个评估
    
    Returns:
        dict: 包含决策信息
    """
    
    if not Path(csv_file).exists():
        return {
            'decision': 'CANNOT_DECIDE',
            'reason': f'CSV 文件不存在: {csv_file}',
            'continue_training': True
        }
    
    # 读取 CSV
    rows = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    if not rows:
        return {
            'decision': 'CANNOT_DECIDE',
            'reason': 'CSV 文件为空',
            'continue_training': True
        }
    
    # 按 step 排序
    rows.sort(key=lambda x: int(x['step']))
    
    # 提取指标值
    metric_values = []
    steps = []
    for row in rows[-window_size:]:  # 只看最后 window_size 个
        try:
            metric_values.append(float(row[metric]))
            steps.append(int(row['step']))
        except (KeyError, ValueError) as e:
            return {
                'decision': 'CANNOT_DECIDE',
                'reason': f'解析 CSV 失败: {e}',
                'continue_training': True
            }
    
    if len(metric_values) < window_size:
        return {
            'decision': 'INSUFFICIENT_DATA',
            'reason': f'评估数据不足 (<{window_size})',
            'continue_training': True,
            'evaluated_steps': steps,
            'metric_values': metric_values
        }
    
    # 计算改进率
    improvements = []
    for i in range(1, len(metric_values)):
        prev_val = metric_values[i-1]
        curr_val = metric_values[i]
        
        # 如果指标越低越好（RMSE），则改进为正
        # 改进率 = (prev - curr) / prev
        improvement_rate = (prev_val - curr_val) / prev_val if prev_val != 0 else 0
        improvements.append(improvement_rate)
    
    # 检查最后的改进
    last_improvement = improvements[-1]
    avg_improvement = np.mean(improvements)
    
    # 决策逻辑
    if last_improvement < improvement_threshold:
        if avg_improvement < improvement_threshold:
            decision = 'STOP'
            reason = f'收敛: 最后改进 {last_improvement*100:.2f}% < 阈值 {improvement_threshold*100:.1f}%'
            continue_training = False
        else:
            decision = 'CONTINUE'
            reason = f'有改进: 平均 {avg_improvement*100:.2f}% >= 阈值 {improvement_threshold*100:.1f}%，但最后改进只有 {last_improvement*100:.2f}%'
            continue_training = True
    else:
        decision = 'CONTINUE'
        reason = f'显著改进: {last_improvement*100:.2f}% >= 阈值 {improvement_threshold*100:.1f}%'
        continue_training = True
    
    return {
        'decision': decision,
        'reason': reason,
        'continue_training': continue_training,
        'metric': metric,
        'evaluated_steps': steps,
        'metric_values': metric_values,
        'improvements': improvements,
        'last_improvement': last_improvement,
        'avg_improvement': avg_improvement,
        'threshold': improvement_threshold
    }

if __name__ == '__main__':
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description='Check convergence from evaluation CSV')
    parser.add_argument('--csv', type=str, required=True, help='Evaluation CSV file')
    parser.add_argument('--metric', type=str, default='f_rmse', 
                       choices=['e_rmse_per_atom', 'f_rmse', 'f_rmse_tail'],
                       help='Metric to check for convergence')
    parser.add_argument('--threshold', type=float, default=0.01, help='Improvement threshold (0.01 = 1%)')
    parser.add_argument('--window', type=int, default=3, help='Number of latest checkpoints to analyze')
    
    args = parser.parse_args()
    
    result = check_convergence(
        args.csv,
        metric=args.metric,
        improvement_threshold=args.threshold,
        window_size=args.window
    )
    
    print("\n" + "=" * 70)
    print("🎯 收敛性分析")
    print("=" * 70)
    print(f"决策: {result['decision']}")
    print(f"原因: {result['reason']}")
    
    if 'evaluated_steps' in result:
        print(f"\n评估的 steps: {result['evaluated_steps']}")
        print(f"对应的 {result['metric']} 值: {result['metric_values']}")
        
        if 'improvements' in result:
            print(f"\n逐步改进率:")
            for i, imp in enumerate(result['improvements']):
                print(f"  Step {result['evaluated_steps'][i]} → {result['evaluated_steps'][i+1]}: {imp*100:+.2f}%")
            
            print(f"\n统计:")
            print(f"  最后改进: {result['last_improvement']*100:+.2f}%")
            print(f"  平均改进: {result['avg_improvement']*100:+.2f}%")
            print(f"  阈值: {result['threshold']*100:.1f}%")
    
    print("\n" + "=" * 70)
    if result['continue_training']:
        print("✓ 建议: 继续训练")
    else:
        print("⊘ 建议: 停止训练")
    print("=" * 70)
    
    # JSON 输出用于脚本处理
    print(json.dumps({
        'decision': result['decision'],
        'continue_training': result['continue_training'],
        'metric': result.get('metric', 'unknown'),
        'last_improvement': float(result.get('last_improvement', 0)),
        'avg_improvement': float(result.get('avg_improvement', 0))
    }))
