# 文档整理日志

日期: 2025-12-28

## 整理策略
将临时报告、分析记录、状态总结等非必须文档移动到 `useless/reports/`，保留核心使用文档与指南。

## 保留的核心文档（根目录）

### 主文档
- `README.md` - 项目主文档
- `README_reproduce.md` - 复现指南
- `README_OPTIMIZED_TRAINING.md` - 训练优化指南
- `README_CUDA_QUICK_START.txt` - CUDA快速开始

### 索引与参考
- `INDEX.md` - 文件索引
- `QUICK_REFERENCE.md` - 快速参考
- `QUICK_START_OPTIMIZATION.md` - 优化快速开始

### 使用指南
- `TROUBLESHOOTING_NEW_DEVICE.md` - 故障排除
- `USING_EXISTING_PROGRAM.md` - 使用现有程序指南
- `NEW_DEVICE_MD_RDF_GUIDE.md` - 新设备MD/RDF指南
- `CUDA_DEPLOYMENT_GUIDE.md` - CUDA部署指南
- `ENVIRONMENT_CONFIG.md` - 环境配置
- `MD_COMPATIBILITY_GUIDE.md` - MD兼容性指南
- `OPTIMIZATION_QUICK_REFERENCE.md` - 优化快速参考
- `PERFORMANCE_OPTIMIZATION_GUIDE.md` - 性能优化指南

### 依赖清单
- `requirements.txt` - Python依赖

## 已移动到 useless/reports/

### 分析与瓶颈报告
- `BOTTLENECK_ANALYSIS.md` - 瓶颈分析报告
- `PERFORMANCE_ANALYSIS.md` - 性能分析报告
- `RDF_ANALYSIS_REPORT.md` - RDF分析报告

### 实现与优化方案
- `IMPLEMENTATION_SUMMARY.md` - 实现总结
- `COMPLETE_OPTIMIZATION_SOLUTION.md` - 完整优化方案

### 状态记录
- `CURRENT_STATUS.md` - 当前状态
- `PROJECT_STATUS.txt` - 项目状态
- `CUDA_SETUP_COMPLETE.md` - CUDA设置完成状态
- `FULL_TRAINING_RUNNING.md` - 完整训练运行记录
- `TRAINING_RUNNING_SUCCESS.md` - 训练成功记录
- `TRAINING_SESSION_STATUS.md` - 训练会话状态

### 测试与总结
- `TEST_RESULTS_SUMMARY.md` - 测试结果总结
- `NEW_DEVICE_TESTING_SUMMARY.md` - 新设备测试总结
- `MODEL_MD_SUMMARY.md` - 模型MD总结

### 清理与变更日志
- `CLEANUP_REPORT.md` - 清理报告
- `RENAMING_CHANGELOG.md` - 重命名变更日志

### 临时输出文本
- `profile_bottleneck_output.txt` - 瓶颈分析输出
- `quick_bottleneck_results.txt` - 快速瓶颈测试结果

## 数据文件（未移动）
以下 RDF 数据文件保留在根目录供快速访问：
- `quick_demo_rdf_data.txt`
- `quick_md_rdf_rdf_data.txt`
- `rdf_1ps_data.txt`
- `rdf_data.txt`

## 后续建议
- 核心文档可合并或精简：考虑将多个 README 合并为一个主 README，其他作为子文档链接。
- 指南类文档可移至 `docs/` 目录以进一步组织结构。
- 数据文件（`*.txt`, `*.npz`, `*.png`）可按用途分类到独立目录。

