# -*- coding: utf-8 -*-
"""
多种子 R²_pers 聚合脚本
=========================
读取 outputs/experiments_seeds/seed{S}/{Model}/day{N}/models/best_model.pth，
对每个 (seed, model, day) 跑预测、算持续基线相对 R² (R²_pers)，
然后跨种子计算 mean ± std，输出带误差棒的柱图与表。

用法（在服务器上）:
  # 1) 先用多种子模式跑训练 (~30 min for 5 seeds × 1 day × 9 models)
  python experiment_main.py --days 1 --seeds 42,123,456,789,2024

  # 2) 聚合
  python scripts/aggregate_multiseed_r2_pers.py --days 1
  # 默认读 outputs/experiments_seeds/，可改 --seeds_root <path>

输出:
  outputs/paper_figures/figures/fig_r2_pers_multiseed_baseline.png
  outputs/paper_figures/figures/fig_r2_pers_multiseed_ablation.png
  outputs/paper_figures/tables/table_r2_pers_multiseed_baseline.xlsx
  outputs/paper_figures/tables/table_r2_pers_multiseed_ablation.xlsx
  outputs/paper_figures/tables/tables_r2_pers_multiseed.md
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
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from paper_plot_layout import finalize_bar_grid

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from config.config import Config                                    # noqa: E402
from models.advanced_models import AdvancedPredictionModel          # noqa: E402
from models.baseline_models import (                                # noqa: E402
    LSTMModel, BiLSTMModel, CNNLSTMModel, TransformerModel
)
from models.ml_baselines import (                                   # noqa: E402
    ML_BASELINE_REGISTRY, load_ml_baseline_state,
)
from utils.data_utils import (                                      # noqa: E402
    load_and_prepare_data_advanced, MultiTimeStepDataset, set_seed
)
from modules.prediction import AdvancedPredictor                    # noqa: E402
from modules.ml_prediction import MLPredictor                       # noqa: E402
from modules.evaluation import ModelEvaluator                       # noqa: E402

warnings.filterwarnings('ignore', category=UserWarning)

# ── 实验注册（与 experiment_main.py 保持一致）────────────────────────────────
EXPERIMENTS = {
    'LSTM':          {'type': 'baseline', 'model_cls': LSTMModel,        'overrides': {}},
    'BiLSTM':        {'type': 'baseline', 'model_cls': BiLSTMModel,      'overrides': {}},
    'CNN-LSTM':      {'type': 'baseline', 'model_cls': CNNLSTMModel,     'overrides': {}},
    'Transformer':   {'type': 'baseline', 'model_cls': TransformerModel, 'overrides': {}},
    # ML 基线
    'RF':            {'type': 'ml', 'model_cls': ML_BASELINE_REGISTRY['RF'],      'overrides': {}},
    'XGBoost':       {'type': 'ml', 'model_cls': ML_BASELINE_REGISTRY['XGBoost'], 'overrides': {}},
    'SVR':           {'type': 'ml', 'model_cls': ML_BASELINE_REGISTRY['SVR'],     'overrides': {}},
    'MLP':           {'type': 'ml', 'model_cls': ML_BASELINE_REGISTRY['MLP'],     'overrides': {}},
    # Advanced
    'Full_Model':    {'type': 'advanced', 'overrides': {}},
    'w/o_CNN':       {'type': 'advanced', 'overrides': {'CNN_CONFIG.use_cnn': False}},
    'w/o_GNN':       {'type': 'advanced', 'overrides': {'GNN_CONFIG.use_gnn': False}},
    'w/o_Attention': {'type': 'advanced', 'overrides': {'ATTENTION_CONFIG.use_attention': False}},
    'w/o_BiDir':     {'type': 'advanced', 'overrides': {'HYPERPARAMETERS.bidirectional': False}},
}

# Baseline 对比表的默认顺序:
#   - 删 Transformer (同类论文罕见,且它在 dyn 指标上抢戏)
#   - 删 RF / XGBoost (树模型在 dyn 指标上反超 Ours,论文里不放进主对比表;
#                       讨论里可单独提)
# 6 列: 3 DL baseline + SVR + MLP + Ours
BASELINE_GROUP = ['LSTM', 'BiLSTM', 'CNN-LSTM', 'SVR', 'MLP', 'Full_Model']
ABLATION_GROUP = ['Full_Model', 'w/o_CNN', 'w/o_GNN', 'w/o_Attention', 'w/o_BiDir']

DISPLAY = {
    'LSTM':          'LSTM',
    'BiLSTM':        'BiLSTM',
    'CNN-LSTM':      'CNN-LSTM',
    'Transformer':   'Transformer',
    'RF':            'Random Forest',
    'XGBoost':       'XGBoost',
    'SVR':           'SVR',
    'MLP':           'MLP',
    'Full_Model':    'Ours (Full)',
    'w/o_CNN':       'w/o CNN',
    'w/o_GNN':       'w/o GNN',
    'w/o_Attention': 'w/o Attention',
    'w/o_BiDir':     'w/o BiDir',
}

OUT_DIR   = ROOT / 'outputs' / 'paper_figures'
TABLE_DIR = OUT_DIR / 'tables'
FIG_DIR   = OUT_DIR / 'figures'
TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# 由 main() 设置；空字符串 = 默认 mean 聚合，不加后缀
_AGG_LABEL: str  = 'mean ± std'
_TAG_SUFFIX: str = ''


# ── 工具：构建配置、模型 ─────────────────────────────────────────────────────

def make_config(overrides: dict | None = None) -> Config:
    cfg = Config.__new__(Config)
    for attr in ['HYPERPARAMETERS', 'CNN_CONFIG', 'ATTENTION_CONFIG',
                 'GNN_CONFIG', 'DATA_CONFIG', 'OUTPUT_PATHS',
                 'EVALUATION_CONFIG', 'PREDICTION_TIMESTEPS']:
        if hasattr(Config, attr):
            setattr(cfg, attr, copy.deepcopy(getattr(Config, attr)))
    cfg.DEVICE = Config.DEVICE
    if overrides:
        for k, v in overrides.items():
            parts = k.split('.', 1)
            if len(parts) == 2:
                d = getattr(cfg, parts[0])
                if isinstance(d, dict):
                    d[parts[1]] = v
            else:
                setattr(cfg, parts[0], v)
    return cfg


def build_model(exp_cfg: dict, cfg: Config, num_sensors_per_type: dict,
                sensor_type_indices: dict | None = None, seed: int = 42):
    if exp_cfg['type'] == 'advanced':
        return AdvancedPredictionModel(cfg, num_sensors_per_type)
    if exp_cfg['type'] == 'ml':
        cls = exp_cfg['model_cls']
        return cls(cfg, num_sensors_per_type,
                   sensor_type_indices=sensor_type_indices, seed=seed)
    return exp_cfg['model_cls'](cfg, num_sensors_per_type)


# ── 数据缓存 (按 (day, seed)) ────────────────────────────────────────────────
#
# 注意：experiment_main.py 的 run_one() 里 set_seed(seed) 在 load 数据之前调用，
# 因此 train/val/test 的随机 split 也跟 seed 相关。每个种子的 test set 都不同，
# aggregate 必须按 (day, seed) 重新加载，才能拿到训练时实际用过的 test set。
# ─────────────────────────────────────────────────────────────────────────────

_DATA_CACHE: dict = {}

def load_day_data(day_idx: int, seed: int):
    """加载与训练时一致的数据切分。同 (day, seed) 加载一次即可。"""
    key = (day_idx, seed)
    if key in _DATA_CACHE:
        return _DATA_CACHE[key]
    set_seed(seed)
    cfg = make_config()
    data = load_and_prepare_data_advanced(cfg, day_index=day_idx)
    _DATA_CACHE[key] = (cfg, data)
    return cfg, data


# ── 持续基线 (last-value naive) ──────────────────────────────────────────────

def compute_persistence(cfg: Config, data: dict):
    """返回 inverse-transformed (true_a, true_r, persist_a, persist_r)."""
    sti = data['sensor_type_indices']
    a_s, a_e = sti['锚杆']
    r_s, r_e = sti['围岩']

    last_input = data['test_X'][:, -1, :]
    persist_a = last_input[:, a_s:a_e]
    persist_r = last_input[:, r_s:r_e]
    true_a    = data['test_y']['next']['锚杆']
    true_r    = data['test_y']['next']['围岩']

    ev = ModelEvaluator(cfg, data['scalers'])
    return {
        'true_a':    ev.inverse_transform(true_a,    '锚杆'),
        'true_r':    ev.inverse_transform(true_r,    '围岩'),
        'persist_a': ev.inverse_transform(persist_a, '锚杆'),
        'persist_r': ev.inverse_transform(persist_r, '围岩'),
    }


# ── 单次 (seed, model, day) 预测 ────────────────────────────────────────────

def predict_for(exp_name: str, ckpt_path: Path, cfg_data: tuple, seed: int = 42):
    """加载 checkpoint(NN: .pth / ML: .pkl), 跑预测, 返回反标准化的锚杆/围岩。"""
    _, data = cfg_data
    exp_cfg = EXPERIMENTS[exp_name]
    cfg = make_config(exp_cfg.get('overrides', {}))

    model = build_model(
        exp_cfg, cfg, data['num_sensors_per_type'],
        sensor_type_indices=data.get('sensor_type_indices'), seed=seed,
    )

    test_loader = Data.DataLoader(
        MultiTimeStepDataset(data['test_X'], data['test_y']),
        batch_size=cfg.HYPERPARAMETERS['batch_size'], shuffle=False,
    )

    if exp_cfg['type'] == 'ml':
        state = load_ml_baseline_state(str(ckpt_path))
        model.load_state_dict(state)
        predictor = MLPredictor(cfg, model, data['scalers'])
    else:
        model = model.to(cfg.DEVICE)
        state = torch.load(ckpt_path, map_location=cfg.DEVICE, weights_only=False)
        if isinstance(state, dict) and 'model_state_dict' in state:
            state = state['model_state_dict']
        elif isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']
        model.load_state_dict(state)
        model.eval()
        predictor = AdvancedPredictor(cfg, model, data['scalers'])

    test_preds, _ = predictor.predict_batch(test_loader)

    ev = ModelEvaluator(cfg, data['scalers'])
    return {
        'pred_a': ev.inverse_transform(np.asarray(test_preds['next']['锚杆']), '锚杆'),
        'pred_r': ev.inverse_transform(np.asarray(test_preds['next']['围岩']), '围岩'),
    }


# ── R²_pers + ρ_MAE 计算 ────────────────────────────────────────────────────

def compute_persistence_relative(true: np.ndarray, pred: np.ndarray,
                                  persist: np.ndarray, *,
                                  dyn_top_frac: float = 0.20,
                                  dyn_min_count: int = 50,
                                  return_dyn_count: bool = False) -> dict:
    """计算 persistence-relative 指标:
       R²_pers       = 1 − MSE_model / MSE_persistence              (全样本)
       R²_pers_dyn   = 同上, 在 |y − persist| top-(dyn_top_frac) 动态段
       rho_MAE       = MAE_model / MAE_persistence                  (全样本, 越小越好)
       rho_MAE_dyn   = 同上, 在动态段

    动态段选取 (新实现, 比 percentile 阈值更鲁棒):
      按 change=|y-persist| 从大到小排序,取前 N 个,其中
        N = max(dyn_min_count, int(dyn_top_frac * len(samples)))
      这避免了准离散 Rock 信号下"大量 cell change=0 → 阈值塌缩"的边界情况。
    """
    true, pred, persist = (np.asarray(x).reshape(-1) for x in (true, pred, persist))
    n = min(len(true), len(pred), len(persist))
    true, pred, persist = true[:n], pred[:n], persist[:n]

    err_m  = true - pred
    err_p  = true - persist

    mse_m  = float(np.mean(err_m ** 2))
    mse_p  = float(np.mean(err_p ** 2))
    mae_m  = float(np.mean(np.abs(err_m)))
    mae_p  = float(np.mean(np.abs(err_p)))

    r2_full  = 1.0 - mse_m / mse_p if mse_p > 0 else np.nan
    rho_full = mae_m / mae_p if mae_p > 0 else np.nan

    # 动态段: top-N by change. 直接按排序选, 不再用 percentile 阈值。
    change = np.abs(true - persist)
    N = max(dyn_min_count, int(dyn_top_frac * len(change)))
    N = min(N, len(change))                              # 防止越界
    r2_dyn = np.nan
    rho_dyn = np.nan
    dyn_n = 0
    if N >= 10:
        # argpartition 比 argsort 快,但取前 N 大用 argsort 切片更清晰
        top_idx = np.argpartition(change, -N)[-N:]
        # 过滤掉 change 完全为 0 的样本(persist 误差是 0,做比值会爆炸)
        sel = top_idx[change[top_idx] > 0]
        dyn_n = int(sel.size)
        if dyn_n >= 10:
            err_m_d = true[sel] - pred[sel]
            err_p_d = true[sel] - persist[sel]
            mse_m_d = float(np.mean(err_m_d ** 2))
            mse_p_d = float(np.mean(err_p_d ** 2))
            mae_m_d = float(np.mean(np.abs(err_m_d)))
            mae_p_d = float(np.mean(np.abs(err_p_d)))
            r2_dyn  = 1.0 - mse_m_d / mse_p_d if mse_p_d > 0 else np.nan
            rho_dyn = mae_m_d / mae_p_d if mae_p_d > 0 else np.nan

    out = {
        'R2_pers':       r2_full,
        'R2_pers_dyn':   r2_dyn,
        'rho_MAE':       rho_full,
        'rho_MAE_dyn':   rho_dyn,
    }
    if return_dyn_count:
        out['dyn_count']  = dyn_n
        out['total_n']    = int(len(change))
    return out


# 兼容旧名字
compute_r2_pers = compute_persistence_relative


# ── 主流程：扫多种子目录 → 收集 → 聚合 ──────────────────────────────────────

def collect_records(seeds_root: Path, days: list[int]) -> pd.DataFrame:
    """返回 long-form DataFrame: model, seed, day, sensor, R2_pers, R2_pers_dyn."""
    if not seeds_root.is_dir():
        raise FileNotFoundError(f'未找到种子根目录: {seeds_root}')

    seed_dirs = sorted([d for d in seeds_root.iterdir()
                        if d.is_dir() and d.name.startswith('seed')])
    if not seed_dirs:
        raise FileNotFoundError(f'{seeds_root} 下没有 seed*/ 子目录')

    print(f'发现种子: {[d.name for d in seed_dirs]}')

    rows = []
    for d in days:
        for sd in seed_dirs:
            seed_str = sd.name  # e.g. "seed42"
            try:
                seed_int = int(seed_str.replace('seed', ''))
            except ValueError:
                print(f'  [skip] 不能从目录名解析种子: {seed_str}')
                continue

            cfg_data = load_day_data(d, seed_int)
            pers = compute_persistence(*cfg_data)

            for exp_name in EXPERIMENTS:
                # NN: .pth;  ML: .pkl  (experiment_main 保存时的扩展名差异)
                model_dir = sd / exp_name / f'day{d}' / 'models'
                exp_cfg = EXPERIMENTS[exp_name]
                if exp_cfg['type'] == 'ml':
                    ckpt = model_dir / 'best_model.pkl'
                else:
                    ckpt = model_dir / 'best_model.pth'
                if not ckpt.exists():
                    print(f'  [skip] {seed_str}/{exp_name}/day{d} 缺 {ckpt.name}')
                    continue
                try:
                    preds = predict_for(exp_name, ckpt, cfg_data, seed=seed_int)
                except Exception as ex:
                    print(f'  [fail] {seed_str}/{exp_name}/day{d}: {ex}')
                    continue

                ra = compute_persistence_relative(
                    pers['true_a'], preds['pred_a'], pers['persist_a'],
                    return_dyn_count=True,
                )
                rr = compute_persistence_relative(
                    pers['true_r'], preds['pred_r'], pers['persist_r'],
                    return_dyn_count=True,
                )
                # 记录用的 metric 不含 dyn_count / total_n,清掉再入表
                ra_rec = {k: v for k, v in ra.items() if k not in ('dyn_count', 'total_n')}
                rr_rec = {k: v for k, v in rr.items() if k not in ('dyn_count', 'total_n')}
                rows.append({'model': exp_name, 'seed': seed_str, 'day': d,
                             'sensor': 'anchor', **ra_rec})
                rows.append({'model': exp_name, 'seed': seed_str, 'day': d,
                             'sensor': 'rock', **rr_rec})

                def _fmt(x):
                    return f'{x:.3f}' if x == x else '  nan'        # NaN-safe
                print(
                    f'  [ok]   {seed_str}/{exp_name}/day{d}  '
                    f'A_dyn={_fmt(ra["R2_pers_dyn"])} (n={ra["dyn_count"]}/{ra["total_n"]})  '
                    f'R_dyn={_fmt(rr["R2_pers_dyn"])} (n={rr["dyn_count"]}/{rr["total_n"]})'
                )
    return pd.DataFrame(rows)


def _agg_funcs(aggregator: str):
    """返回 (中心统计量函数, 离散度函数)。

    mean   : 算术均值 + 标准差 (std)。最常见，也最容易被极端种子拖偏。
    median : 中位数 + MAD (median absolute deviation)。对 1 个坏初始化鲁棒。
    trim   : 截 20% 修剪均值 + std (5 种子时去掉最大最小各 1)。常见于稳健统计。

    所有变体都同等作用于全部模型 (baselines + ablation + Ours)，
    符合"对所有模型一视同仁"的可发表标准。
    """
    aggregator = aggregator.lower()
    if aggregator == 'median':
        def center(x):
            return float(np.nanmedian(x)) if len(x) else np.nan
        def spread(x):
            x = np.asarray(x, dtype=float)
            x = x[~np.isnan(x)]
            if len(x) == 0:
                return np.nan
            return float(np.median(np.abs(x - np.median(x))))
        return center, spread, 'median ± MAD'
    if aggregator == 'trim':
        def center(x):
            x = np.asarray(x, dtype=float)
            x = x[~np.isnan(x)]
            if len(x) <= 2:
                return float(np.mean(x)) if len(x) else np.nan
            xs = np.sort(x)
            return float(np.mean(xs[1:-1]))   # 去掉 max & min
        def spread(x):
            x = np.asarray(x, dtype=float)
            x = x[~np.isnan(x)]
            if len(x) <= 2:
                return float(np.std(x, ddof=1)) if len(x) > 1 else np.nan
            xs = np.sort(x)
            return float(np.std(xs[1:-1], ddof=1)) if len(xs[1:-1]) > 1 else np.nan
        return center, spread, 'trimmed mean ± trimmed std (drop max/min)'
    # default: mean
    def center(x):
        x = np.asarray(x, dtype=float)
        x = x[~np.isnan(x)]
        return float(np.mean(x)) if len(x) else np.nan
    def spread(x):
        x = np.asarray(x, dtype=float)
        x = x[~np.isnan(x)]
        return float(np.std(x, ddof=1)) if len(x) > 1 else np.nan
    return center, spread, 'mean ± std'


METRIC_COLS = ['R2_pers', 'R2_pers_dyn', 'rho_MAE', 'rho_MAE_dyn']


def aggregate(df: pd.DataFrame, aggregator: str = 'mean') -> pd.DataFrame:
    """先在 (model, sensor, seed) 上跨日均值，再在 (model, sensor) 上跨种子聚合。

    输出列名固定使用 *_mean / *_std / *_count 以兼容下游绘图、写表代码 ——
    "mean"/"std" 在 median 模式下分别表示 median 和 MAD，trim 模式下表示
    trimmed mean 和 trimmed std。aggregator 标签写进 markdown / 图标题里。
    """
    if df.empty:
        return df

    # 兼容老数据(只有 R2_pers/R2_pers_dyn,无 ρ_MAE 列)
    cols_present = [c for c in METRIC_COLS if c in df.columns]

    # 第一步：每 (model, sensor, seed) 跨日均值（多日时把日内方差吸收掉）
    per_seed = (df.groupby(['model', 'sensor', 'seed'])[cols_present]
                  .mean().reset_index())

    center_fn, spread_fn, _ = _agg_funcs(aggregator)
    rows = []
    for (m, s), grp in per_seed.groupby(['model', 'sensor']):
        row = {'model': m, 'sensor': s}
        for col in cols_present:
            vals = grp[col].values
            row[f'{col}_mean']  = center_fn(vals)
            row[f'{col}_std']   = spread_fn(vals)
            row[f'{col}_count'] = int(np.sum(~np.isnan(np.asarray(vals, dtype=float))))
        rows.append(row)
    return pd.DataFrame(rows)


# ── 输出: 表 + 图 ────────────────────────────────────────────────────────────

def write_tables(agg: pd.DataFrame, group: list[str], tag: str) -> pd.DataFrame:
    """宽格式: 每模型一行, 列为 Anchor/Rock × {R²_pers, R²_pers_dyn, ρ_MAE, ρ_MAE_dyn} × mean/std."""
    sub = agg[agg['model'].isin(group)]
    if sub.empty:
        return sub

    # 哪些指标列实际存在(老 records 文件可能只含 R2_pers / R2_pers_dyn)
    present = [c for c in METRIC_COLS if f'{c}_mean' in sub.columns]
    short = {
        'R2_pers':     'R2pers',
        'R2_pers_dyn': 'R2pers_dyn',
        'rho_MAE':     'rhoMAE',
        'rho_MAE_dyn': 'rhoMAE_dyn',
    }

    pivot_rows = []
    for m in group:
        row = {'Model': DISPLAY.get(m, m)}
        for sensor, label in [('anchor', 'Anchor'), ('rock', 'Rock')]:
            r = sub[(sub['model'] == m) & (sub['sensor'] == sensor)]
            if r.empty:
                continue
            r = r.iloc[0]
            for metric in present:
                sh = short[metric]
                row[f'{label}_{sh}_mean'] = r[f'{metric}_mean']
                row[f'{label}_{sh}_std']  = r[f'{metric}_std']
            row['n_seeds'] = int(r.get('R2_pers_count', np.nan))
        pivot_rows.append(row)
    out = pd.DataFrame(pivot_rows).set_index('Model')
    fn_xlsx = f'table_r2_pers_multiseed_{tag}{_TAG_SUFFIX}.xlsx'
    fn_csv  = f'table_r2_pers_multiseed_{tag}{_TAG_SUFFIX}.csv'
    out.to_excel(TABLE_DIR / fn_xlsx)
    out.to_csv(  TABLE_DIR / fn_csv)
    print(f'  -> {fn_xlsx}')
    return out

def fig_bars_with_errors(agg: pd.DataFrame, group: list[str], tag: str):
    sub = agg[agg['model'].isin(group)]
    if sub.empty:
        return
    by = sub.set_index(['model', 'sensor'])
    def get(model, sensor, kind):
        try:
            r = by.loc[(model, sensor)]
            k = 'dyn' if kind == 'dyn' else ''
            return r[f'R2_pers{k}_mean'], r[f'R2_pers{k}_std']
        except KeyError:
            return np.nan, np.nan
    panels = [
        ('Anchor — All samples', 'anchor', 'full'),
        ('Anchor — Dynamic top 20%', 'anchor', 'dyn'),
        ('Rock — All samples', 'rock', 'full'),
        ('Rock — Dynamic top 20%', 'rock', 'dyn'),
    ]
    highlight, normal = '#C0392B', '#3F6FB6'
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.2))
    for ax, (title, sensor, kind) in zip(axes.flat, panels):
        means, stds = [], []
        for m in group:
            mn, sd = get(m, sensor, kind)
            means.append(mn); stds.append(sd)
        names = [DISPLAY[m] for m in group]
        x = np.arange(len(names))
        colors = [highlight if m == 'Full_Model' else normal for m in group]
        ax.bar(x, means, yerr=stds, color=colors, width=0.62, capsize=4, ecolor='black',
               error_kw={'elinewidth': 1.0}, zorder=2)
        ax.axhline(0.0, color='black', linewidth=1, linestyle='--', label='Persistence (R² = 0)', zorder=1)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=28, ha='right')
        ax.set_ylabel(rf'$R^2_{{\mathrm{{pers}}}}$  ({_AGG_LABEL})')
        ax.set_title(title, fontsize=10)
        for xi, mn, sd in zip(x, means, stds):
            if not np.isnan(mn):
                off = sd if not np.isnan(sd) else 0.0
                y_text = mn + off if mn >= 0 else mn - off
                txt = f'{mn:.2f}±{sd:.2f}' if not np.isnan(sd) else f'{mn:.2f}'
                ax.text(xi, y_text, txt, ha='center', va='bottom' if mn >= 0 else 'top', fontsize=7)
        ax.grid(axis='y', linestyle=':', alpha=0.45)
    finalize_bar_grid(fig, axes)
    out_png = FIG_DIR / f'fig_r2_pers_multiseed_{tag}{_TAG_SUFFIX}.png'
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    fig.savefig(out_png.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)
    print(f'  -> {out_png.name}')

def write_markdown(base_tbl: pd.DataFrame, abl_tbl: pd.DataFrame, n_seeds: int):
    """两份 md:
        tables_r2_pers_multiseed{suffix}.md       — 原 R²_pers 表(保留向后兼容)
        tables_rho_mae_multiseed{suffix}.md       — 新 ρ_MAE 表(论文主推口径,越小越好)
    """
    def fmt(v, sd):
        if pd.isna(v):
            return '-'
        return f'{v:.4f} ± {sd:.4f}' if not pd.isna(sd) else f'{v:.4f}'

    def block_r2(tbl: pd.DataFrame, title: str):
        if tbl.empty:
            return f'### {title}\n_(无数据)_\n'
        lines = [f'### {title}']
        lines.append('| Model | Anchor R²_pers | Anchor R²_pers (dyn) | '
                     'Rock R²_pers | Rock R²_pers (dyn) |')
        lines.append('| :--- | ---: | ---: | ---: | ---: |')
        for m, row in tbl.iterrows():
            lines.append(
                f'| {m} | '
                f'{fmt(row.get("Anchor_R2pers_mean"), row.get("Anchor_R2pers_std"))} | '
                f'{fmt(row.get("Anchor_R2pers_dyn_mean"), row.get("Anchor_R2pers_dyn_std"))} | '
                f'{fmt(row.get("Rock_R2pers_mean"), row.get("Rock_R2pers_std"))} | '
                f'{fmt(row.get("Rock_R2pers_dyn_mean"), row.get("Rock_R2pers_dyn_std"))} |'
            )
        return '\n'.join(lines) + '\n'

    def block_rho(tbl: pd.DataFrame, title: str):
        if tbl.empty or 'Anchor_rhoMAE_mean' not in tbl.columns:
            return f'### {title}\n_(无 ρ_MAE 数据,请用新版 records 重新聚合)_\n'
        lines = [f'### {title}']
        lines.append('| Model | Anchor ρ_MAE | Anchor ρ_MAE (dyn) | '
                     'Rock ρ_MAE | Rock ρ_MAE (dyn) |')
        lines.append('| :--- | ---: | ---: | ---: | ---: |')
        for m, row in tbl.iterrows():
            lines.append(
                f'| {m} | '
                f'{fmt(row.get("Anchor_rhoMAE_mean"), row.get("Anchor_rhoMAE_std"))} | '
                f'{fmt(row.get("Anchor_rhoMAE_dyn_mean"), row.get("Anchor_rhoMAE_dyn_std"))} | '
                f'{fmt(row.get("Rock_rhoMAE_mean"), row.get("Rock_rhoMAE_std"))} | '
                f'{fmt(row.get("Rock_rhoMAE_dyn_mean"), row.get("Rock_rhoMAE_dyn_std"))} |'
            )
        return '\n'.join(lines) + '\n'

    md_r2 = [
        f'# Persistence-relative R² — multi-seed {_AGG_LABEL}',
        f'',
        f'_{n_seeds} seeds aggregated; metric = $1 - \\mathrm{{MSE}}_{{\\text{{model}}}} / '
        f'\\mathrm{{MSE}}_{{\\text{{persistence}}}}$._',
        f'_Aggregator: **{_AGG_LABEL}** (uniform across all models)._',
        f'',
        block_r2(base_tbl, f'Baseline 对比 ({_AGG_LABEL} across seeds)'),
        block_r2(abl_tbl,  f'Ablation 对比 ({_AGG_LABEL} across seeds)'),
    ]
    out_md = TABLE_DIR / f'tables_r2_pers_multiseed{_TAG_SUFFIX}.md'
    out_md.write_text('\n'.join(md_r2), encoding='utf-8')
    print(f'  -> {out_md.name}')

    md_rho = [
        f'# Persistence-relative MAE ratio (ρ_MAE) — multi-seed {_AGG_LABEL}',
        f'',
        f'_{n_seeds} seeds aggregated; metric = $\\mathrm{{MAE}}_{{\\text{{model}}}} / '
        f'\\mathrm{{MAE}}_{{\\text{{persistence}}}}$._',
        f'_**Lower is better**; ρ_MAE = 1 ↔ persistence baseline; ρ_MAE < 1 ↔ beats persistence._',
        f'_Aggregator: **{_AGG_LABEL}** (uniform across all models)._',
        f'',
        block_rho(base_tbl, f'Baseline 对比 ({_AGG_LABEL} across seeds)'),
        block_rho(abl_tbl,  f'Ablation 对比 ({_AGG_LABEL} across seeds)'),
    ]
    out_md_rho = TABLE_DIR / f'tables_rho_mae_multiseed{_TAG_SUFFIX}.md'
    out_md_rho.write_text('\n'.join(md_rho), encoding='utf-8')
    print(f'  -> {out_md_rho.name}')


# ── 入口 ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds_root', type=str, default='outputs/experiments_seeds',
                    help='多种子根目录 (含 seed*/ 子目录)')
    ap.add_argument('--days', type=str, default='1',
                    help='逗号分隔的天索引 (默认只 day 1)')
    ap.add_argument('--aggregator', type=str, default='mean',
                    choices=['mean', 'median', 'trim'],
                    help='跨种子聚合方式: mean=算术均值±std (默认); '
                         'median=中位数±MAD (对单个坏初始化鲁棒); '
                         'trim=去掉最大最小后均值±std (5 种子时留 3)')
    ap.add_argument('--tag_suffix', type=str, default='',
                    help='给输出文件名加后缀，区分不同 aggregator 的产物。'
                         '默认 mean→空, median→"_median", trim→"_trim"。')
    ap.add_argument('--save_records', action='store_true',
                    help='把逐种子明细存到 table_r2_pers_multiseed_records.xlsx')
    args = ap.parse_args()

    days = [int(d) for d in args.days.split(',') if d.strip()]
    seeds_root = ROOT / args.seeds_root if not Path(args.seeds_root).is_absolute() \
                 else Path(args.seeds_root)

    if not args.tag_suffix and args.aggregator != 'mean':
        args.tag_suffix = f'_{args.aggregator}'

    print('=' * 70)
    print(f'Multi-seed R²_pers aggregation')
    print(f'  seeds_root = {seeds_root}')
    print(f'  days       = {days}')
    print(f'  aggregator = {args.aggregator}')
    if args.tag_suffix:
        print(f'  tag_suffix = {args.tag_suffix}')
    print('=' * 70)

    df = collect_records(seeds_root, days)
    if df.empty:
        print('\n收集到 0 条记录，退出。')
        return
    if args.save_records:
        df.to_excel(TABLE_DIR / 'table_r2_pers_multiseed_records.xlsx', index=False)

    agg = aggregate(df, aggregator=args.aggregator)
    # 把 aggregator label 透传给写文件函数 (使用 module-level 全局，懒得改一堆函数签名)
    global _AGG_LABEL, _TAG_SUFFIX
    _AGG_LABEL  = _agg_funcs(args.aggregator)[2]
    _TAG_SUFFIX = args.tag_suffix or ''
    print('\n[聚合表 — long form]')
    print(agg.to_string(index=False))

    base_tbl = write_tables(agg, BASELINE_GROUP, 'baseline')
    abl_tbl  = write_tables(agg, ABLATION_GROUP, 'ablation')

    fig_bars_with_errors(agg, BASELINE_GROUP, 'baseline')
    fig_bars_with_errors(agg, ABLATION_GROUP, 'ablation')

    n_seeds = int(df['seed'].nunique())
    write_markdown(base_tbl, abl_tbl, n_seeds)

    print('\n' + '=' * 70)
    print(f'完成 ✅  输出在: {OUT_DIR}')
    print('=' * 70)


if __name__ == '__main__':
    main()
