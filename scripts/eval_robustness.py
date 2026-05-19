# -*- coding: utf-8 -*-
"""
鲁棒性评估 + 统计显著性检验 + 时序响应分析
============================================
用法（服务器上执行）:
    python scripts/eval_robustness.py --day 1 --seed 42
    python scripts/eval_robustness.py --day 1 --seed 42,123,456,789,2024  # 多 seed

输出:
    outputs/paper_figures/robustness/
        noise_mae_curves.png/pdf          — 噪声鲁棒性曲线
        missing_mae_curves.png/pdf        — 缺失数据鲁棒性曲线
        temporal_response.png/pdf         — 时序响应对比
        table_robustness.xlsx/md          — 鲁棒性汇总表
        table_statistical_significance.xlsx/md — Wilcoxon 检验结果
"""

import argparse
import copy
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.utils.data as Data
from matplotlib import pyplot as plt
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from config.config import Config
from models.advanced_models import AdvancedPredictionModel
from models.baseline_models import LSTMModel, BiLSTMModel, CNNLSTMModel
from models.ml_baselines import ML_BASELINE_REGISTRY, load_ml_baseline_state
from utils.data_utils import load_and_prepare_data_advanced, MultiTimeStepDataset, set_seed
from modules.prediction import AdvancedPredictor
from modules.ml_prediction import MLPredictor
from modules.evaluation import ModelEvaluator
from scripts.aggregate_multiseed_r2_pers import (
    EXPERIMENTS, make_config, build_model,
    load_day_data, compute_persistence, predict_for,
    compute_persistence_relative,
)

warnings.filterwarnings('ignore')

# ── 输出目录 ──────────────────────────────────────────────────────────────────
OUT_DIR = ROOT / 'outputs' / 'paper_figures' / 'robustness'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 要评估的模型 ──────────────────────────────────────────────────────────────
ROBUST_MODELS = ['LSTM', 'BiLSTM', 'CNN-LSTM', 'SVR', 'MLP', 'Full_Model']
DISPLAY = {
    'LSTM':'LSTM','BiLSTM':'Bi-LSTM','CNN-LSTM':'CNN-LSTM',
    'SVR':'SVR','MLP':'MLP','Full_Model':'Ours',
}

NOISE_LEVELS = [0.1, 0.2, 0.5]      # σ multiplier × per-channel std
MISSING_RATES = [0.10, 0.20, 0.30]  # fraction of time steps masked


# ═══════════════════════════════════════════════════════════════════════════════
#  1. NOISE INJECTION
# ═══════════════════════════════════════════════════════════════════════════════

def add_noise(test_X: np.ndarray, noise_std: float, seed: int = 999):
    """对归一化后的 test_X 逐通道加高斯噪声。
    noise_std = multiplier × per-channel std (在归一化空间中).
    """
    rng = np.random.RandomState(seed)
    X = test_X.copy()                      # (N, T, C)
    # 沿 (N, T) 维度计算每个通道的 std
    ch_std = np.std(X, axis=(0, 1), keepdims=True)  # (1, 1, C)
    noise = rng.randn(*X.shape) * noise_std * ch_std
    return X + noise


def eval_noise(seeds_root: Path, models: list[str], day: int,
               seeds: list[int], noise_levels: list[float]) -> pd.DataFrame:
    """对每个 (seed, model, noise_level) 组合跑预测, 返回长格式 DataFrame."""
    rows = []
    for sd in seeds:
        seed_dir = seeds_root / f'seed{sd}'
        cfg_data = load_day_data(day, sd)
        pers = compute_persistence(*cfg_data)

        for model_name in models:
            exp_cfg = EXPERIMENTS[model_name]
            model_dir = seed_dir / model_name / f'day{day}' / 'models'
            ckpt = (model_dir / 'best_model.pkl') if exp_cfg['type'] == 'ml' \
                   else (model_dir / 'best_model.pth')
            if not ckpt.exists():
                print(f'  [skip] seed{sd}/{model_name}/day{day}')
                continue

            # 干净数据预测 (baseline)
            try:
                preds_clean = predict_for(model_name, ckpt, cfg_data, seed=sd)
            except Exception as e:
                print(f'  [fail] clean seed{sd}/{model_name}: {e}')
                continue

            for nl in noise_levels:
                _, data_orig = cfg_data
                # 对归一化后的 test_X 加噪声
                noisy_X = add_noise(data_orig['test_X'], nl)
                # 重建 DataLoader
                noisy_data = copy.deepcopy(data_orig)
                noisy_data['test_X'] = noisy_X
                cfg_noisy = make_config(exp_cfg.get('overrides', {}))

                try:
                    model = build_model(
                        exp_cfg, cfg_noisy, data_orig['num_sensors_per_type'],
                        sensor_type_indices=data_orig.get('sensor_type_indices'), seed=sd)
                    test_loader = Data.DataLoader(
                        MultiTimeStepDataset(noisy_X, data_orig['test_y']),
                        batch_size=cfg_noisy.HYPERPARAMETERS['batch_size'], shuffle=False)

                    if exp_cfg['type'] == 'ml':
                        state = load_ml_baseline_state(str(ckpt))
                        model.load_state_dict(state)
                        predictor = MLPredictor(cfg_noisy, model, data_orig['scalers'])
                    else:
                        model = model.to(cfg_noisy.DEVICE)
                        state = torch.load(ckpt, map_location=cfg_noisy.DEVICE, weights_only=False)
                        if isinstance(state, dict) and 'model_state_dict' in state:
                            state = state['model_state_dict']
                        elif isinstance(state, dict) and 'state_dict' in state:
                            state = state['state_dict']
                        model.load_state_dict(state)
                        model.eval()
                        predictor = AdvancedPredictor(cfg_noisy, model, data_orig['scalers'])

                    test_preds, _ = predictor.predict_batch(test_loader)
                except Exception as e:
                    print(f'  [fail] noise={nl} seed{sd}/{model_name}: {e}')
                    continue

                ev = ModelEvaluator(cfg_noisy, data_orig['scalers'])
                preds_noisy = {
                    'pred_a': ev.inverse_transform(np.asarray(test_preds['next']['锚杆']), '锚杆'),
                    'pred_r': ev.inverse_transform(np.asarray(test_preds['next']['围岩']), '围岩'),
                }

                r_a = compute_persistence_relative(pers['true_a'], preds_noisy['pred_a'], pers['persist_a'])
                r_r = compute_persistence_relative(pers['true_r'], preds_noisy['pred_r'], pers['persist_r'])

                for sensor, r in [('anchor', r_a), ('rock', r_r)]:
                    rows.append({
                        'model': model_name, 'seed': sd, 'day': day,
                        'noise_level': nl, 'sensor': sensor,
                        'R2_pers': r['R2_pers'], 'R2_pers_dyn': r['R2_pers_dyn'],
                        'rho_MAE': r['rho_MAE'], 'rho_MAE_dyn': r['rho_MAE_dyn'],
                    })
                print(f'  [ok] noise={nl} seed{sd}/{model_name}: A_dyn={r_a["R2_pers_dyn"]:.4f} R_dyn={r_r["R2_pers_dyn"]:.4f}')

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
#  2. MISSING DATA
# ═══════════════════════════════════════════════════════════════════════════════

def mask_and_interpolate(test_X: np.ndarray, missing_rate: float, seed: int = 999):
    """随机 mask missing_rate 的时间步, 用线性插值补全。
    test_X: (N, T, C) 归一化后的数据.
    返回补全后的 test_X.
    """
    rng = np.random.RandomState(seed)
    X = test_X.copy()
    N, T, C = X.shape
    # 对每个样本, 随机 mask 一些时间步
    mask = rng.rand(N, T) < missing_rate
    # 不 mask 第一个和最后一个时间步 (保证插值有边界)
    mask[:, 0] = False
    mask[:, -1] = False
    for n in range(N):
        for c in range(C):
            masked_idx = np.where(mask[n])[0]
            if len(masked_idx) == 0:
                continue
            valid_idx = np.where(~mask[n])[0]
            X[n, masked_idx, c] = np.interp(masked_idx, valid_idx, X[n, valid_idx, c])
    return X


def eval_missing(seeds_root: Path, models: list[str], day: int,
                 seeds: list[int], missing_rates: list[float]) -> pd.DataFrame:
    """对每个 (seed, model, missing_rate) 组合跑预测."""
    rows = []
    for sd in seeds:
        seed_dir = seeds_root / f'seed{sd}'
        cfg_data = load_day_data(day, sd)
        pers = compute_persistence(*cfg_data)

        for model_name in models:
            exp_cfg = EXPERIMENTS[model_name]
            model_dir = seed_dir / model_name / f'day{day}' / 'models'
            ckpt = (model_dir / 'best_model.pkl') if exp_cfg['type'] == 'ml' \
                   else (model_dir / 'best_model.pth')
            if not ckpt.exists():
                continue

            for mr in missing_rates:
                _, data_orig = cfg_data
                masked_X = mask_and_interpolate(data_orig['test_X'], mr)
                m_data = copy.deepcopy(data_orig)
                m_data['test_X'] = masked_X
                cfg_m = make_config(exp_cfg.get('overrides', {}))

                try:
                    model = build_model(
                        exp_cfg, cfg_m, data_orig['num_sensors_per_type'],
                        sensor_type_indices=data_orig.get('sensor_type_indices'), seed=sd)
                    test_loader = Data.DataLoader(
                        MultiTimeStepDataset(masked_X, data_orig['test_y']),
                        batch_size=cfg_m.HYPERPARAMETERS['batch_size'], shuffle=False)

                    if exp_cfg['type'] == 'ml':
                        state = load_ml_baseline_state(str(ckpt))
                        model.load_state_dict(state)
                        predictor = MLPredictor(cfg_m, model, data_orig['scalers'])
                    else:
                        model = model.to(cfg_m.DEVICE)
                        state = torch.load(ckpt, map_location=cfg_m.DEVICE, weights_only=False)
                        if isinstance(state, dict) and 'model_state_dict' in state:
                            state = state['model_state_dict']
                        elif isinstance(state, dict) and 'state_dict' in state:
                            state = state['state_dict']
                        model.load_state_dict(state)
                        model.eval()
                        predictor = AdvancedPredictor(cfg_m, model, data_orig['scalers'])

                    test_preds, _ = predictor.predict_batch(test_loader)
                except Exception as e:
                    print(f'  [fail] missing={mr} seed{sd}/{model_name}: {e}')
                    continue

                ev = ModelEvaluator(cfg_m, data_orig['scalers'])
                preds_m = {
                    'pred_a': ev.inverse_transform(np.asarray(test_preds['next']['锚杆']), '锚杆'),
                    'pred_r': ev.inverse_transform(np.asarray(test_preds['next']['围岩']), '围岩'),
                }

                r_a = compute_persistence_relative(pers['true_a'], preds_m['pred_a'], pers['persist_a'])
                r_r = compute_persistence_relative(pers['true_r'], preds_m['pred_r'], pers['persist_r'])

                for sensor, r in [('anchor', r_a), ('rock', r_r)]:
                    rows.append({
                        'model': model_name, 'seed': sd, 'day': day,
                        'missing_rate': mr, 'sensor': sensor,
                        'R2_pers': r['R2_pers'], 'R2_pers_dyn': r['R2_pers_dyn'],
                        'rho_MAE': r['rho_MAE'], 'rho_MAE_dyn': r['rho_MAE_dyn'],
                    })
                print(f'  [ok] missing={mr} seed{sd}/{model_name}: A_dyn={r_a["R2_pers_dyn"]:.4f} R_dyn={r_r["R2_pers_dyn"]:.4f}')

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
#  3. TEMPORAL RESPONSE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def detect_major_transitions(true: np.ndarray, top_n: int = 10,
                              min_gap: int = 30, window_radius: int = 40):
    """检测 top-N 个最大的突变事件 (按 |y[t+1] - y[t]| 峰值排序).
    返回 [(center_idx, peak_magnitude), ...] 列表.
    """
    change = np.abs(true[1:] - true[:-1])
    smoothed = np.convolve(change, np.ones(min_gap) / min_gap, mode='same')
    peaks = []
    for i in range(window_radius, len(smoothed) - window_radius):
        if smoothed[i] == np.max(smoothed[i - window_radius:i + window_radius]):
            peaks.append((i, change[i]))
    peaks.sort(key=lambda x: x[1], reverse=True)
    return peaks[:top_n]


def compute_cross_correlation_lag(true: np.ndarray, pred: np.ndarray,
                                   centers: list, window_radius: int = 40,
                                   max_lag: int = 15) -> dict:
    """用互相关计算每个突变事件处预测的相位延迟.
    正 lag = 预测滞后于真实.
    """
    lags = []
    correlations = []
    for center, _ in centers:
        t_start = max(0, center - window_radius)
        t_end = min(len(true), center + window_radius)
        if t_end - t_start < 2 * max_lag:
            continue
        t_win = true[t_start:t_end] - np.mean(true[t_start:t_end])
        p_win = pred[t_start:t_end] - np.mean(pred[t_start:t_end])
        best_lag = 0
        best_corr = -np.inf
        for lag in range(-max_lag, max_lag + 1):
            if lag >= 0:
                t_a, p_a = t_win[lag:], p_win[:len(t_win)-lag]
            else:
                t_a, p_a = t_win[:lag], p_win[-lag:]
            if len(t_a) < 5:
                continue
            corr = np.corrcoef(t_a, p_a)[0, 1]
            if not np.isnan(corr) and corr > best_corr:
                best_corr = corr
                best_lag = lag
        if best_corr > -np.inf:
            lags.append(best_lag)
            correlations.append(best_corr)
    if not lags:
        return {'mean_lag': np.nan, 'median_lag': np.nan, 'std_lag': np.nan,
                'mean_corr': np.nan, 'n_events': len(centers), 'n_valid': 0}
    return {
        'mean_lag': float(np.mean(lags)),
        'median_lag': float(np.median(lags)),
        'std_lag': float(np.std(lags)),
        'mean_corr': float(np.mean(correlations)),
        'n_events': len(centers),
        'n_valid': len(lags),
    }


def eval_temporal_response(seeds_root: Path, models: list[str], day: int, seed: int):
    """评估各模型对突变事件的时序响应速度."""
    results = []
    seed_dir = seeds_root / f'seed{seed}'
    cfg_data = load_day_data(day, seed)
    pers = compute_persistence(*cfg_data)
    true_a = pers['true_a'].reshape(-1)

    # 检测 top-10 突变事件 (互相关法)
    centers = detect_major_transitions(true_a, top_n=10)
    print(f'  检测到 {len(centers)} 个主要突变事件')

    for model_name in models:
        exp_cfg = EXPERIMENTS[model_name]
        model_dir = seed_dir / model_name / f'day{day}' / 'models'
        ckpt = (model_dir / 'best_model.pkl') if exp_cfg['type'] == 'ml' \
               else (model_dir / 'best_model.pth')
        if not ckpt.exists():
            continue
        try:
            preds = predict_for(model_name, ckpt, cfg_data, seed=seed)
        except Exception as e:
            print(f'  [fail] temporal seed{seed}/{model_name}: {e}')
            continue

        pred_a_flat = preds['pred_a'].reshape(-1)
        lag_info = compute_cross_correlation_lag(true_a, pred_a_flat, centers)
        lag_info['model'] = model_name
        lag_info['seed'] = seed
        lag_info['day'] = day
        results.append(lag_info)
        print(f'  [ok] {model_name}: median_lag={lag_info["median_lag"]:.1f} steps, '
              f'corr={lag_info["mean_corr"]:.3f} (n={lag_info["n_valid"]})')

    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════════════════════
#  4. 统计显著性检验
# ═══════════════════════════════════════════════════════════════════════════════

def eval_statistical_significance(seeds_root: Path, days: list[int],
                                  seeds: list[int]) -> pd.DataFrame:
    """Wilcoxon signed-rank test: Ours vs each baseline, 对多 seed 跨日 R²_pers_dyn 配对检验."""
    # 复用 aggregate_multiseed_r2_pers 的 collect_records
    from scripts.aggregate_multiseed_r2_pers import collect_records
    df = collect_records(seeds_root, days)
    if df.empty:
        print('WARNING: collect_records 返回空, 统计检验跳过')
        return pd.DataFrame()

    # 对每个 (model, sensor) 取跨日均值
    per_seed = df.groupby(['model', 'sensor', 'seed'])[['R2_pers', 'R2_pers_dyn', 'rho_MAE', 'rho_MAE_dyn']].mean()

    rows = []
    baselines = ['LSTM', 'BiLSTM', 'CNN-LSTM', 'SVR', 'MLP']
    ours = 'Full_Model'
    metrics = ['R2_pers', 'R2_pers_dyn', 'rho_MAE', 'rho_MAE_dyn']

    for sensor in ['anchor', 'rock']:
        ours_vals = {}
        for m in metrics:
            try:
                ours_vals[m] = per_seed.loc[(ours, sensor, slice(None)), m].values
            except KeyError:
                ours_vals[m] = np.array([])

        for bl in baselines:
            row = {'sensor': sensor, 'baseline': bl}
            for m in metrics:
                try:
                    bl_vals = per_seed.loc[(bl, sensor, slice(None)), m].values
                    if len(bl_vals) >= 5 and len(ours_vals.get(m, [])) >= 5:
                        # 对齐长度
                        n = min(len(bl_vals), len(ours_vals[m]))
                        diff = bl_vals[:n] - ours_vals[m][:n]
                        if np.all(diff == 0):
                            row[f'{m}_p'] = 1.0
                            row[f'{m}_significant'] = False
                        else:
                            stat, p = wilcoxon(bl_vals[:n], ours_vals[m][:n])
                            row[f'{m}_p'] = p
                            row[f'{m}_significant'] = p < 0.05
                    else:
                        row[f'{m}_p'] = np.nan
                        row[f'{m}_significant'] = False
                except (KeyError, ValueError) as e:
                    row[f'{m}_p'] = np.nan
                    row[f'{m}_significant'] = False
            rows.append(row)

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
#  5. 绘图
# ═══════════════════════════════════════════════════════════════════════════════

def plot_robustness_curves(df: pd.DataFrame, x_col: str, x_label: str,
                           title_prefix: str, filename: str):
    """画鲁棒性曲线: x=扰动程度, y=R²_pers_dyn (Anchor + Rock 各一 panel)."""
    if df.empty:
        print(f'  空数据, 跳过 {filename}')
        return

    # 聚合: median across seeds
    agg = df.groupby(['model', 'sensor', x_col])['R2_pers_dyn'].median().reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    colors = {'LSTM':'#7F8C8D','BiLSTM':'#95A5A6','CNN-LSTM':'#3498DB',
              'SVR':'#E67E22','MLP':'#9B59B6','Full_Model':'#C0392B'}
    linestyles = {'Full_Model': '-', 'SVR': '--', 'CNN-LSTM': '-.',
                  'LSTM': ':', 'BiLSTM': ':', 'MLP': '--'}

    for ax, sensor, label in [(axes[0], 'anchor', 'Anchor-bolt'),
                               (axes[1], 'rock', 'Surrounding-rock')]:
        sub = agg[agg['sensor'] == sensor]
        for model in ROBUST_MODELS:
            mdata = sub[sub['model'] == model]
            if not mdata.empty:
                xs = mdata[x_col].values
                ys = mdata['R2_pers_dyn'].values
                sort_idx = np.argsort(xs)
                ax.plot(np.array(xs)[sort_idx], np.array(ys)[sort_idx],
                        color=colors[model], linestyle=linestyles.get(model, '-'),
                        marker='o', markersize=6, linewidth=2,
                        label=DISPLAY[model])
        ax.axhline(0, color='black', linestyle='--', linewidth=0.8, label='Persistence (R²=0)')
        ax.set_xlabel(x_label)
        ax.set_ylabel('R²_pers (dynamic subset)')
        ax.set_title(f'{label}', fontsize=11)
        ax.legend(fontsize=7, ncol=2, loc='upper center',
                  bbox_to_anchor=(0.5, -0.18), frameon=False)
        ax.grid(axis='y', linestyle=':', alpha=0.4)

    fig.suptitle(f'{title_prefix}: Robustness of disturbance prediction under {x_label.lower()}',
                 fontsize=12, y=1.02)
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    out = OUT_DIR / filename
    fig.savefig(out.with_suffix('.png'), dpi=300, bbox_inches='tight')
    fig.savefig(out.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)
    print(f'  -> {out.name}')


def plot_temporal_response(df: pd.DataFrame):
    """画时序响应延迟对比图."""
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    models_ordered = ['LSTM', 'BiLSTM', 'CNN-LSTM', 'SVR', 'MLP', 'Full_Model']
    colors = ['#7F8C8D', '#95A5A6', '#3498DB', '#E67E22', '#9B59B6', '#C0392B']

    names = []
    medians = []
    for i, m in enumerate(models_ordered):
        sub = df[df['model'] == m]
        if not sub.empty:
            names.append(DISPLAY[m])
            medians.append(sub['median_lag'].iloc[0])
            ax.bar(i, sub['median_lag'].iloc[0], color=colors[i], width=0.55,
                   yerr=sub['std_lag'].iloc[0] if 'std_lag' in sub.columns else 0,
                   capsize=4)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20)
    ax.set_ylabel('Median response lag (time steps)', fontsize=11)
    ax.set_title('Temporal response to structural transitions (lower = faster)', fontsize=11)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.grid(axis='y', linestyle=':', alpha=0.4)
    fig.tight_layout()
    out = OUT_DIR / 'temporal_response.png'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    fig.savefig(out.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)
    print(f'  -> {out.name}')


# ═══════════════════════════════════════════════════════════════════════════════
#  6. 主入口
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument('--root', type=str, default='outputs/experiments_seeds')
    ap.add_argument('--day', type=int, default=1)
    ap.add_argument('--seed', type=str, default='42',
                    help='逗号分隔, 默认 42 (单 seed 快速验证)')
    ap.add_argument('--days_significance', type=str, default='1,2,3,4,5,6,7',
                    help='统计检验用的天范围')
    ap.add_argument('--skip_noise', action='store_true')
    ap.add_argument('--skip_missing', action='store_true')
    ap.add_argument('--skip_temporal', action='store_true')
    ap.add_argument('--skip_significance', action='store_true')
    args = ap.parse_args()

    seeds_root = ROOT / args.root if not Path(args.root).is_absolute() \
                 else Path(args.root)
    seeds_list = [int(s.strip()) for s in args.seed.split(',') if s.strip()]
    days_list = [int(d.strip()) for d in args.days_significance.split(',') if d.strip()]

    print('=' * 70)
    print(f'Robustness + Temporal + Statistical evaluation')
    print(f'  root = {seeds_root}')
    print(f'  day = {args.day}, seeds = {seeds_list}')
    print('=' * 70)

    all_dfs = {}

    # ── Noise ──
    if not args.skip_noise:
        print('\n[1/4] Noise injection robustness...')
        df_noise = eval_noise(seeds_root, ROBUST_MODELS, args.day,
                              seeds_list, NOISE_LEVELS)
        if not df_noise.empty:
            all_dfs['noise'] = df_noise
            plot_robustness_curves(df_noise, 'noise_level',
                                   'Noise level (σ multiplier)',
                                   'Noise Robustness', 'noise_mae_curves')
            df_noise.to_csv(OUT_DIR / 'raw_noise.csv', index=False)
            print(f'  Saved {len(df_noise)} records')
        else:
            print('  WARNING: no noise results collected')

    # ── Missing ──
    if not args.skip_missing:
        print('\n[2/4] Missing data robustness...')
        df_missing = eval_missing(seeds_root, ROBUST_MODELS, args.day,
                                  seeds_list, MISSING_RATES)
        if not df_missing.empty:
            all_dfs['missing'] = df_missing
            plot_robustness_curves(df_missing, 'missing_rate',
                                   'Missing data rate',
                                   'Missing Data Robustness', 'missing_mae_curves')
            df_missing.to_csv(OUT_DIR / 'raw_missing.csv', index=False)
            print(f'  Saved {len(df_missing)} records')
        else:
            print('  WARNING: no missing data results collected')

    # ── Temporal ──
    if not args.skip_temporal:
        print('\n[3/4] Temporal response analysis...')
        df_temporal = eval_temporal_response(seeds_root, ROBUST_MODELS, args.day, seeds_list[0])
        if not df_temporal.empty:
            all_dfs['temporal'] = df_temporal
            plot_temporal_response(df_temporal)
            df_temporal.to_csv(OUT_DIR / 'raw_temporal.csv', index=False)
            print(f'  Saved {len(df_temporal)} models')
        else:
            print('  WARNING: no temporal results collected')

    # ── Statistical ──
    if not args.skip_significance:
        print('\n[4/4] Statistical significance (Wilcoxon)...')
        df_sig = eval_statistical_significance(seeds_root, days_list, seeds_list)
        if not df_sig.empty:
            all_dfs['significance'] = df_sig
            out_xlsx = OUT_DIR / 'table_statistical_significance.xlsx'
            df_sig.to_excel(out_xlsx, index=False)
            print(f'  -> {out_xlsx.name}')

            # markdown
            md = ['# Statistical significance (Wilcoxon signed-rank, Ours vs baseline)',
                  '', '| Sensor | Baseline | R²_pers_dyn p | Significant (p<0.05) | ρ_MAE_dyn p | Significant |',
                  '| :--- | :--- | ---: | :---: | ---: | :---: |']
            for _, row in df_sig.iterrows():
                md.append(
                    f'| {row["sensor"]} | {row["baseline"]} | '
                    f'{row.get("R2_pers_dyn_p", 1):.4f} | {row.get("R2_pers_dyn_significant", False)} | '
                    f'{row.get("rho_MAE_dyn_p", 1):.4f} | {row.get("rho_MAE_dyn_significant", False)} |')
            (OUT_DIR / 'table_statistical_significance.md').write_text('\n'.join(md), encoding='utf-8')
            print(f'  -> table_statistical_significance.md')
        else:
            print('  WARNING: statistical test produced no results')

    # ── Summary table ──
    print('\n' + '=' * 70)
    print(f'Done. Output: {OUT_DIR}')
    print('=' * 70)


if __name__ == '__main__':
    main()
