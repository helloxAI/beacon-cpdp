# BEACON Reproduction Code

## Environment

- Python 3.14 (verified on Windows 11); CPU is sufficient
- Install dependencies:

```bash
pip install -r requirements.txt
```

## Data and Black-Box Models

- Datasets: `Datasets/{PROMISE,NASA,AEEEM,JIRA}/*.arff`
- Pre-trained black-box models: `black_model/baseline/models/<pair_dir>/*.joblib`
  (18 backbones x 2 training pairs; load and reuse directly, no training required)

## Usage

```bash
# Smoke test: rq1.1 scenario, TNB backbone only (~2 minutes serially)
python run_experiments.py --scenario smoke

# Run a specific scenario with selected backbones in parallel
python run_experiments.py --scenario rq1.1 --models SupCon_DP,HDP_KS --jobs 4

# Sampling demo (runs a small slice of each scenario)
python run_sample.py --jobs 8

# Full reproduction (main experiments + ablation + sensitivity)
python run_experiments.py --scenario all --jobs 12

# Generate summary tables from existing results (output to results/tables/)
python make_tables.py
```

- `--scenario` options: `smoke`, `rq1.1`, `rq1.2`, `rq2.1`, `rq2.2`, `rq2.3`
  (the five above constitute `main`), `rq4`, `rq5`, `all`
- `--models`: comma-separated backbone names; defaults to all 18
- `--jobs`: number of parallel processes; `--data-root`: dataset directory (default `Datasets`)
- Results are appended row by row to `results/raw/<scenario>.csv`; rerunning after an interruption automatically skips completed rows
- Caches are stored in `results/cache/` (target features + black-box outputs) and are automatically reused across runs
