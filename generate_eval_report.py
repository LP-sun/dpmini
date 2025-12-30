#!/usr/bin/env python3
"""
固定验证集评估报告生成器
对比多个模型的性能
"""

import json
from pathlib import Path
import numpy as np

def generate_report():
    """生成对比报告"""
    
    # 加载评估结果
    metrics_path = Path('logs/val_metrics.json')
    with open(metrics_path, 'r') as f:
        current_metrics = json.load(f)
    
    # 读取备份 README 中的基准指标
    backup_readme = Path('backups/best_checkpoints/README.txt')
    
    report = []
    report.append("=" * 80)
    report.append("固定验证集评估报告 (FIXED VALIDATION SET EVALUATION REPORT)")
    report.append("=" * 80)
    
    report.append(f"\n【评估时间】{Path('val_idx.npy').stat().st_mtime}")
    report.append(f"【验证集大小】200 帧（固定，来自 collect/O64H128，种子=42）")
    report.append(f"【当前模型】checkpoints_optimized_v3/model_step80000.pt")
    
    report.append("\n" + "─" * 80)
    report.append("当前评估结果 (CURRENT EVALUATION)")
    report.append("─" * 80)
    
    report.append(f"\n力预测性能 (Force Prediction):")
    report.append(f"  F_RMSE_mean:     {current_metrics['f_rmse_mean']:.6f} eV/Å")
    report.append(f"  F_RMSE_median:   {current_metrics['f_rmse_median']:.6f} eV/Å")
    report.append(f"  F_RMSE_95%ile:   {current_metrics['f_rmse_95']:.6f} eV/Å")
    report.append(f"  F_RMSE_tail10%:  {current_metrics['f_rmse_tail']:.6f} eV/Å（最差10%）")
    report.append(f"  F_RMSE_std:      {current_metrics['f_rmse_std']:.6f} eV/Å")
    
    report.append(f"\n能量预测性能 (Energy Prediction):")
    report.append(f"  E_RMSE/atom:     {current_metrics['e_rmse_per_atom']:.6f} eV")
    
    report.append("\n" + "─" * 80)
    report.append("基准对比 (BASELINE COMPARISON)")
    report.append("─" * 80)
    
    # 从 README 提取基准数据
    baseline_f_rmse = 0.715  # from README
    baseline_f_rmse_tail = 1.288  # from README
    baseline_e_rmse = 61.22  # from README
    
    # 计算改进
    f_rmse_improvement = (baseline_f_rmse - current_metrics['f_rmse_mean']) / baseline_f_rmse * 100
    f_rmse_tail_improvement = (baseline_f_rmse_tail - current_metrics['f_rmse_tail']) / baseline_f_rmse_tail * 100
    e_rmse_improvement = (baseline_e_rmse - current_metrics['e_rmse_per_atom']) / baseline_e_rmse * 100
    
    report.append(f"\n基准值（V3 原始训练，Step 80k）:")
    report.append(f"  F_RMSE_mean:     0.715000 eV/Å")
    report.append(f"  F_RMSE_tail:     1.288000 eV/Å")
    report.append(f"  E_RMSE/atom:     61.22000 eV")
    
    report.append(f"\n当前模型对比:")
    report.append(f"  F_RMSE_mean:     {current_metrics['f_rmse_mean']:.6f} eV/Å "
                 f"({f_rmse_improvement:+.2f}% vs 基准)")
    report.append(f"  F_RMSE_tail:     {current_metrics['f_rmse_tail']:.6f} eV/Å "
                 f"({f_rmse_tail_improvement:+.2f}% vs 基准)")
    report.append(f"  E_RMSE/atom:     {current_metrics['e_rmse_per_atom']:.6f} eV "
                 f"({e_rmse_improvement:+.2f}% vs 基准)")
    
    report.append("\n" + "─" * 80)
    report.append("性能评估 (PERFORMANCE ASSESSMENT)")
    report.append("─" * 80)
    
    if f_rmse_improvement < 0.5:
        status = "⚠️  略优于基准"
    elif f_rmse_improvement < -2:
        status = "❌ 性能下降（需调查）"
    else:
        status = "✓ 与基准相当或更优"
    
    report.append(f"\nF_RMSE 指标: {status}")
    
    report.append(f"\n详细分析:")
    report.append(f"  1. 力预测平均误差: {current_metrics['f_rmse_mean']:.4f} eV/Å")
    if current_metrics['f_rmse_mean'] < baseline_f_rmse:
        report.append(f"     → 比基准低 {(baseline_f_rmse - current_metrics['f_rmse_mean'])*100/(baseline_f_rmse):.1f}%")
    else:
        report.append(f"     → 比基准高 {(current_metrics['f_rmse_mean'] - baseline_f_rmse)*100/(baseline_f_rmse):.1f}%")
    
    report.append(f"\n  2. 困难样本表现: {current_metrics['f_rmse_tail']:.4f} eV/Å (tail 10%)")
    if current_metrics['f_rmse_tail'] < baseline_f_rmse_tail:
        report.append(f"     → 比基准低 {(baseline_f_rmse_tail - current_metrics['f_rmse_tail'])*100/(baseline_f_rmse_tail):.1f}%")
    else:
        report.append(f"     → 比基准高 {(current_metrics['f_rmse_tail'] - baseline_f_rmse_tail)*100/(baseline_f_rmse_tail):.1f}%")
    
    report.append(f"\n  3. 能量预测: {current_metrics['e_rmse_per_atom']:.4f} eV/atom")
    if current_metrics['e_rmse_per_atom'] < baseline_e_rmse:
        report.append(f"     → 比基准低 {(baseline_e_rmse - current_metrics['e_rmse_per_atom'])*100/(baseline_e_rmse):.1f}%")
    else:
        report.append(f"     → 比基准高 {(current_metrics['e_rmse_per_atom'] - baseline_e_rmse)*100/(baseline_e_rmse):.1f}%")
    
    report.append("\n" + "─" * 80)
    report.append("建议 (RECOMMENDATIONS)")
    report.append("─" * 80)
    
    # 生成建议
    avg_improvement = (f_rmse_improvement + f_rmse_tail_improvement) / 2
    
    if avg_improvement > 2:
        report.append(f"\n✅ 模型性能已显著改进 ({avg_improvement:.1f}% 平均改进)")
        report.append("   建议: 继续使用这个模型，考虑进一步微调")
    elif avg_improvement > 0:
        report.append(f"\n✓ 模型性能略有改进 ({avg_improvement:.1f}% 平均改进)")
        report.append("   建议: 模型表现稳定，可继续训练或使用")
    elif avg_improvement > -2:
        report.append(f"\n⚠️  模型性能基本相同 ({avg_improvement:.1f}% 平均变化)")
        report.append("   建议: 性能与基准相当，继续训练可能无显著收益")
        report.append("      可考虑停止训练以节省资源")
    else:
        report.append(f"\n❌ 模型性能有下降 ({avg_improvement:.1f}% 平均下降)")
        report.append("   建议: 需要调查问题原因")
        report.append("      检查: 1) 训练配置是否正确")
        report.append("           2) 数据加载是否正常")
        report.append("           3) 超参数是否合适")
    
    report.append("\n" + "=" * 80)
    report.append("关键指标总结 (KEY METRICS SUMMARY)")
    report.append("=" * 80)
    
    report.append(f"\n✓ F_RMSE_mean:    {current_metrics['f_rmse_mean']:.4f} eV/Å (基准: 0.7150)")
    report.append(f"✓ F_RMSE_tail:    {current_metrics['f_rmse_tail']:.4f} eV/Å (基准: 1.2880)")
    report.append(f"✓ E_RMSE/atom:    {current_metrics['e_rmse_per_atom']:.4f} eV    (基准: 61.2200)")
    report.append(f"\n评估样本数: {current_metrics['n_frames']} 帧")
    
    # 保存报告
    report_text = '\n'.join(report)
    report_path = Path('logs/FIXED_VALSET_EVALUATION_REPORT.txt')
    report_path.write_text(report_text)
    
    print(report_text)
    print(f"\n📄 报告已保存: {report_path}")
    
    return report_text

if __name__ == '__main__':
    generate_report()
