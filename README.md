# Roof Warning System: CNN-BiLSTM-Attention-GNN

A deep learning framework for underground mine roof displacement prediction using a hybrid CNN-BiLSTM-Attention-GNN architecture.

## Architecture

The full model (`Full_Model`) stacks:
- **CNN**: local feature extraction from multi-sensor sequences
- **BiLSTM**: bidirectional temporal modeling
- **Attention**: adaptive weighting of time steps
- **GNN**: spatial correlation between anchor bolt and rock sensors

## Repository Structure

```
├── config/             Model and training hyperparameters
├── models/             Model definitions (advanced + baselines)
│   ├── advanced_models.py    Full CNN-BiLSTM-Attention-GNN model
│   ├── baseline_models.py    LSTM / BiLSTM / CNN-LSTM / Transformer
│   └── ml_baselines.py       SVR / MLP (sklearn-based)
├── modules/            Training, prediction, evaluation logic
├── utils/              Data loading and visualization utilities
├── scripts/            Figure and table generation scripts
│   ├── generate_paper_figures.py   Main figure pipeline
│   ├── aggregate_multiseed_r2_pers.py  Multi-seed R² aggregation
│   ├── eval_robustness.py          Noise / missing-data robustness
│   ├── gen_table_per_day.py        Per-day result tables
│   ├── paper_plot_layout.py        Shared plot utilities
│   └── redraw_figures.py           Redraw final figures from cache
├── data/               Raw CSV files (not tracked by Git; see data/README.md)
├── results/
│   ├── figures/        Final paper figures (1.png – 5.png)
│   ├── tables/         Result tables (CSV / xlsx / markdown)
│   └── robustness/     Robustness evaluation outputs
├── experiment_main.py  Main training entry point
└── requirements.txt
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Place data files

See `data/README.md` for the expected file layout.

### 3. Run experiments (single seed)

```bash
python experiment_main.py --days 1 --seeds 42
```

### 4. Run multi-seed experiments (5 seeds × 7 days)

```bash
python experiment_main.py --days 1,2,3,4,5,6,7 --seeds 42,123,456,789,2024
```

### 5. Generate paper figures

```bash
python scripts/generate_paper_figures.py --day 1 --variant main
```

## Results

Pre-computed results are in `results/`. Key metric: **Persistence-relative R²** (`R²_pers = 1 − MSE_model / MSE_persistence`), evaluated on both all samples and the dynamic top-20% subset.

| Model | Anchor R²_pers (dyn) | Rock R²_pers (dyn) |
|---|---|---|
| Ours (Full) | **0.17** | **0.30** |
| BiLSTM | 0.00 | 0.00 |
| CNN-LSTM | 0.09 | 0.28 |
| SVR | 0.24 | 0.33 |
| MLP | 0.16 | −0.04 |

> Ours is the **only model that consistently avoids identity collapse** across all sensor types and time windows.

## Citation

*(To be updated after publication)*
