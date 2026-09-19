# BEACON 复现代码

## 环境

- Python 3.14（Windows 11 验证通过），CPU 即可
- 安装依赖：

```bash
pip install -r requirements.txt
```

## 数据与黑盒

- 数据集：`Datasets/{PROMISE,NASA,AEEEM,JIRA}/*.arff`
- 预训练黑盒：`black_model/baseline/models/<pair_dir>/*.joblib`
  （18 个骨干 × 2 个训练对，直接加载复用，无需训练）

## 用法

```bash
# 冒烟：rq1.1 场景、仅 TNB 骨干（串行约 2 分钟）
python run_experiments.py --scenario smoke

# 指定场景与骨干并行
python run_experiments.py --scenario rq1.1 --models SupCon_DP,HDP_KS --jobs 4

# 抽样演示（每个场景各跑一小片）
python run_sample.py --jobs 8

# 全量复现（主实验 + 消融 + 敏感性）
python run_experiments.py --scenario all --jobs 12

# 由已有结果生成汇总表（输出到 results/tables/）
python make_tables.py
```

- `--scenario` 可选：`smoke`、`rq1.1`、`rq1.2`、`rq2.1`、`rq2.2`、`rq2.3`
  （以上五个即 `main`）、`rq4`、`rq5`、`all`
- `--models` 逗号分隔的骨干名，默认全部 18 个
- `--jobs` 并行进程数；`--data-root` 指定数据集目录（默认 `Datasets`）
- 结果按行追加到 `results/raw/<scenario>.csv`，中断后重跑自动跳过已完成行
- 缓存位于 `results/cache/`（目标特征 + 黑盒输出），重复运行自动复用
