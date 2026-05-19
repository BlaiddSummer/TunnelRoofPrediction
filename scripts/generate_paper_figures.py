# -*- coding: utf-8 -*-
"""
论文第五章图表生成脚本
======================
基于 outputs/experiments_seeds/seed{S}/ 下保存的 .pth (DL) / .pkl (ML) 与
experiment_summary.xlsx, 生成对比 / 消融 的图与表; 风格参照 "预期效果图/" 中的样图。

用法:
    python scripts/generate_paper_figures.py                              # 默认 day=1, seed=42, both 套图
    python scripts/generate_paper_figures.py --day 1 --variant main       # 仅 6 列正文版
    python scripts/generate_paper_figures.py --day 1 --variant appendix   # 仅 9 列附录版
    python scripts/generate_paper_figures.py --only tables                # 只生成表
    python scripts/generate_paper_figures.py --skip-multiseed-tables      # 不调聚合脚本

输出:
    outputs/paper_figures/
        ├── main/figures/            正文 6 列图 (png + pdf)
        ├── main/tables/             正文表
        ├── appendix/figures/        附录 9 列图
        ├── appendix/tables/         附录表
        ├── tables/                  多 seed 主表 + Table 4 逐天表 (共享, 不分 variant)
        └── _cache/                  预测缓存 (.npz), 二次运行直接用
"""
import os
import sys
import copy
import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.utils.data as Data
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from paper_plot_layout import finalize_bar_grid, annotate_bar_values
from matplotlib import rcParams
from matplotlib.ticker import MaxNLocator

warnings.filterwarnings('ignore')

# ── 项目根目录 ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from config.config import Config                                     # noqa: E402
from models.advanced_models import AdvancedPredictionModel           # noqa: E402
from models.baseline_models import (                                 # noqa: E402
    LSTMModel, BiLSTMModel, CNNLSTMModel, TransformerModel
)
from models.ml_baselines import (                                    # noqa: E402
    ML_BASELINE_REGISTRY, load_ml_baseline_state
)
from utils.data_utils import (                                       # noqa: E402
    load_and_prepare_data_advanced, MultiTimeStepDataset, set_seed
)
from modules.prediction import AdvancedPredictor                     # noqa: E402
from modules.ml_prediction import MLPredictor                        # noqa: E402
from modules.evaluation import ModelEvaluator                        # noqa: E402

# ── 全局配置 ──────────────────────────────────────────────────────────────────
EXPERIMENTS = {
    'LSTM':          {'type': 'baseline', 'model_cls': LSTMModel,        'overrides': {}},
    'BiLSTM':        {'type': 'baseline', 'model_cls': BiLSTMModel,      'overrides': {}},
    'CNN-LSTM':      {'type': 'baseline', 'model_cls': CNNLSTMModel,     'overrides': {}},
    'Transformer':   {'type': 'baseline', 'model_cls': TransformerModel, 'overrides': {}},
    'SVR':           {'type': 'ml', 'model_cls': ML_BASELINE_REGISTRY['SVR']},
    'MLP':           {'type': 'ml', 'model_cls': ML_BASELINE_REGISTRY['MLP']},
    'RF':            {'type': 'ml', 'model_cls': ML_BASELINE_REGISTRY['RF']},
    'XGBoost':       {'type': 'ml', 'model_cls': ML_BASELINE_REGISTRY['XGBoost']},
    'Full_Model':    {'type': 'advanced', 'overrides': {}},
    'w/o_CNN':       {'type': 'advanced', 'overrides': {'CNN_CONFIG.use_cnn': False}},
    'w/o_GNN':       {'type': 'advanced', 'overrides': {'GNN_CONFIG.use_gnn': False}},
    'w/o_Attention': {'type': 'advanced', 'overrides': {'ATTENTION_CONFIG.use_attention': False}},
    'w/o_BiDir':     {'type': 'advanced', 'overrides': {'HYPERPARAMETERS.bidirectional': False}},
}
# 论文主表 6 列(默认),附录 9 列;两套图各跑一次。
BASELINE_GROUP_MAIN     = ['LSTM', 'BiLSTM', 'CNN-LSTM', 'SVR', 'MLP', 'Full_Model']
BASELINE_GROUP_APPENDIX = ['LSTM', 'BiLSTM', 'CNN-LSTM', 'Transformer',
                           'SVR', 'MLP', 'RF', 'XGBoost', 'Full_Model']
BASELINE_GROUP = BASELINE_GROUP_MAIN     # 运行时由 setup_variant() 切换
ABLATION_GROUP = ['Full_Model', 'w/o_CNN', 'w/o_GNN', 'w/o_Attention', 'w/o_BiDir']

# 实验名 -> 论文中显示名(英文, 与 ./预期效果图 风格一致)
DISPLAY = {
    'LSTM':          'LSTM',
    'BiLSTM':        'BiLSTM',
    'CNN-LSTM':      'CNN-LSTM',
    'Transformer':   'Transformer',
    'SVR':           'SVR',
    'MLP':           'MLP',
    'RF':            'RF',
    'XGBoost':       'XGBoost',
    'Full_Model':    'Ours (Full)',
    'w/o_CNN':       'w/o CNN',
    'w/o_GNN':       'w/o GNN',
    'w/o_Attention': 'w/o Attention',
    'w/o_BiDir':     'w/o BiDir',
}
SENSOR_DISPLAY = {'锚杆': 'Anchor', '围岩': 'Surrounding rock'}
SENSOR_TYPES = ['锚杆', '围岩']
METRICS = ['MAE', 'RMSE', 'MAPE', 'R2']
METRIC_LABELS = {'MAE': 'MAE', 'RMSE': 'RMSE', 'MAPE': 'MAPE (%)', 'R2': 'R²'}

# 配色 (与论文样图接近的低饱和蓝绿调)
PALETTE = ['#3F6FB6', '#7CB7D6', '#5BAE9D', '#E8A87C', '#C38D9E',
           '#85586F', '#4D8B6F', '#B3823F', '#88498F']

# 这些路径默认指向 variant=main, 由 setup_variant() 在 main() 中按需切换。
# CACHE_DIR / SUMMARY_XLS / 多 seed 主表的写入路径不随 variant 改变, 共享在 paper_figures 根下。
ROOT_OUT    = ROOT / 'outputs' / 'paper_figures'
OUT_DIR     = ROOT_OUT / 'main'
TABLE_DIR   = OUT_DIR / 'tables'
FIG_DIR     = OUT_DIR / 'figures'
CACHE_DIR   = ROOT_OUT / '_cache'
# SUMMARY_XLS / EXP_ROOT 由 setup_seed() 设置, 默认 seed42
SEED        = 42
EXP_ROOT    = ROOT / 'outputs' / 'experiments_seeds' / f'seed{SEED}'
SUMMARY_XLS = EXP_ROOT / 'experiment_summary.xlsx'


def setup_variant(variant: str):
    """切换 OUT_DIR / TABLE_DIR / FIG_DIR / BASELINE_GROUP 到指定 variant。

    variant='main' → 6 列基线, 'appendix' → 9 列基线。CACHE_DIR 不变(预测共享缓存)。
    """
    global OUT_DIR, TABLE_DIR, FIG_DIR, BASELINE_GROUP
    OUT_DIR   = ROOT_OUT / variant
    TABLE_DIR = OUT_DIR / 'tables'
    FIG_DIR   = OUT_DIR / 'figures'
    BASELINE_GROUP = (BASELINE_GROUP_MAIN if variant == 'main'
                      else BASELINE_GROUP_APPENDIX)


def setup_seed(seed: int):
    """切换 ckpt 根目录与 SUMMARY_XLS 到指定 seed。"""
    global SEED, EXP_ROOT, SUMMARY_XLS
    SEED        = seed
    EXP_ROOT    = ROOT / 'outputs' / 'experiments_seeds' / f'seed{seed}'
    SUMMARY_XLS = EXP_ROOT / 'experiment_summary.xlsx'


def _ckpt_path(exp_name: str, day_idx: int, seed: int | None = None) -> Path:
    """根据 EXPERIMENTS[exp].type 选择 .pth(DL) 或 .pkl(ML), 返回 ckpt 全路径。

    ML 模型保存为 pickle (best_model.pkl), DL/Advanced 保存为 torch (.pth)。
    """
    s = SEED if seed is None else seed
    spec = EXPERIMENTS.get(exp_name)
    ext = 'pkl' if (spec and spec.get('type') == 'ml') else 'pth'
    return (EXP_ROOT / exp_name / f'day{day_idx}' / 'models' / f'best_model.{ext}')


def setup_matplotlib():
    """统一图表风格。"""
    rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'Microsoft YaHei',
                               'SimHei', 'Times New Roman']
    rcParams['axes.unicode_minus'] = False
    rcParams['axes.labelsize'] = 11
    rcParams['axes.titlesize'] = 12
    rcParams['xtick.labelsize'] = 10
    rcParams['ytick.labelsize'] = 10
    rcParams['legend.fontsize'] = 10
    rcParams['lines.linewidth'] = 1.4
    rcParams['axes.spines.top'] = False
    rcParams['axes.spines.right'] = False
    rcParams['axes.grid'] = True
    rcParams['grid.alpha'] = 0.25
    rcParams['grid.linestyle'] = '--'


# ── 配置 / 模型 / 数据 加载 ────────────────────────────────────────────────────

def make_config(overrides: dict | None = None) -> Config:
    """复制 experiment_main.py 中的隔离 Config 构造。"""
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
            if len(parts) == 2 and isinstance(getattr(cfg, parts[0], None), dict):
                getattr(cfg, parts[0])[parts[1]] = v
            else:
                setattr(cfg, parts[0], v)
    return cfg


def load_data_for_day(day_idx: int):
    """加载指定天的数据 (复用 train/val/test 切分)."""
    set_seed(42)
    cfg = make_config()
    return load_and_prepare_data_advanced(cfg, day_index=day_idx)


def build_and_load(exp_name: str, day_idx: int, num_sensors_per_type: dict,
                   seed: int | None = None):
    """重建模型并加载权重。
    - DL / Advanced: .pth + torch.load
    - ML(SVR/MLP/RF/XGBoost): .pkl + load_ml_baseline_state
    路径统一走 outputs/experiments_seeds/seed{S}/{exp}/day{N}/models/best_model.{pth|pkl}。
    """
    spec = EXPERIMENTS[exp_name]
    cfg = make_config(spec.get('overrides', {}))

    if spec['type'] == 'ml':
        # ML baseline: sensor_type_indices 可以暂置 None, load_state_dict 会从 pickle 恢复。
        model = spec['model_cls'](cfg, num_sensors_per_type,
                                  sensor_type_indices=None, seed=(seed or SEED))
    elif spec['type'] == 'advanced':
        model = AdvancedPredictionModel(cfg, num_sensors_per_type)
    else:
        model = spec['model_cls'](cfg, num_sensors_per_type)

    ckpt_path = _ckpt_path(exp_name, day_idx, seed)
    if not ckpt_path.exists():
        raise FileNotFoundError(f'缺少权重: {ckpt_path}')

    if spec['type'] == 'ml':
        state = load_ml_baseline_state(str(ckpt_path))
        model.load_state_dict(state)
        # ML 模型走 numpy, 不需要 .to() / .eval(), 但接口兼容 nn.Module 调用
        return model, cfg, ckpt_path

    state = torch.load(ckpt_path, map_location=cfg.DEVICE, weights_only=False)
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    model.load_state_dict(state)
    model = model.to(cfg.DEVICE).eval()
    return model, cfg, ckpt_path


def get_predictions(exp_name: str, day_idx: int, force: bool = False,
                    seed: int | None = None):
    """获取测试集预测 (反标准化后). 若有缓存直接返回.

    Cache key 带 seed{S} 前缀, 避免不同 seed 互相污染。
    """
    s = SEED if seed is None else seed
    cache = CACHE_DIR / f'seed{s}_{exp_name.replace("/", "_")}_day{day_idx}.npz'
    if cache.exists() and not force:
        z = np.load(cache, allow_pickle=True)
        return {
            'pred_anchor':  z['pred_anchor'],
            'true_anchor':  z['true_anchor'],
            'pred_rock':    z['pred_rock'],
            'true_rock':    z['true_rock'],
        }

    print(f'  -> 计算预测: {exp_name} day{day_idx} (seed{s})')
    data = load_data_for_day(day_idx)
    model, cfg, _ = build_and_load(exp_name, day_idx,
                                   data['num_sensors_per_type'], seed=s)

    test_loader = Data.DataLoader(
        MultiTimeStepDataset(data['test_X'], data['test_y']),
        batch_size=cfg.HYPERPARAMETERS['batch_size'], shuffle=False
    )
    if EXPERIMENTS[exp_name]['type'] == 'ml':
        predictor = MLPredictor(cfg, model, data['scalers'])
    else:
        predictor = AdvancedPredictor(cfg, model, data['scalers'])
    preds, acts = predictor.predict_batch(test_loader)

    evaluator = ModelEvaluator(cfg, data['scalers'])

    def _denorm(arr, sensor):
        return evaluator.inverse_transform(arr, sensor)

    out = {
        'pred_anchor': _denorm(preds['next']['锚杆'], '锚杆'),
        'true_anchor': _denorm(acts['next']['锚杆'], '锚杆'),
        'pred_rock':   _denorm(preds['next']['围岩'], '围岩'),
        'true_rock':   _denorm(acts['next']['围岩'], '围岩'),
    }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **out)
    return out


def get_history(exp_name: str, day_idx: int):
    """读取完整训练历史（取「最长」一条轨迹）。

    来源: (1) outputs/experiments_seeds/seed{S}/{exp}/day{N}/models/ 下 checkpoint;
         (2) outputs/loss_curve_runs/{exp_name}_day{N}/ 下 history.json 或 latest_checkpoint
    （专用长训脚本的曲线通常比批量实验更长，用于论文 Fig.8）
    ML baseline 无 epoch 训练历史, 直接返回 None。
    """
    if EXPERIMENTS.get(exp_name, {}).get('type') == 'ml':
        return None
    cands: list[dict] = []
    base = EXP_ROOT / exp_name / f'day{day_idx}' / 'models'
    for fname in ('latest_checkpoint.pth', 'best_model.pth'):
        p = base / fname
        if not p.exists():
            continue
        try:
            ck = torch.load(p, map_location='cpu', weights_only=False)
            h = ck.get('history') if isinstance(ck, dict) else None
            if h and h.get('train_loss'):
                cands.append(h)
        except Exception:
            pass

    lcd = ROOT / 'outputs' / 'loss_curve_runs'
    if lcd.is_dir():
        prefix = f'{exp_name}_day{day_idx}'
        for run_dir in sorted(lcd.iterdir()):
            if not run_dir.is_dir():
                continue
            name = run_dir.name
            if name != prefix and not name.startswith(prefix + '_'):
                continue
            hj = run_dir / 'history.json'
            if hj.exists():
                try:
                    data = json.loads(hj.read_text(encoding='utf-8'))
                    if data.get('train_loss'):
                        cands.append({
                            'train_loss': data['train_loss'],
                            'val_loss': data.get('val_loss', []),
                        })
                except Exception:
                    pass
            lp = run_dir / 'models' / 'latest_checkpoint.pth'
            if lp.exists():
                try:
                    ck = torch.load(lp, map_location='cpu', weights_only=False)
                    h = ck.get('history') if isinstance(ck, dict) else None
                    if h and h.get('train_loss'):
                        cands.append(h)
                except Exception:
                    pass

    if not cands:
        return None
    return max(cands, key=lambda h: len(h['train_loss']))


# ── Persistence baseline (last-value naive) ──────────────────────────────────

def get_persistence(day_idx: int, force: bool = False):
    """朴素持续基线: 预测值 = 输入序列最后一帧的值 (denorm)."""
    cache = CACHE_DIR / f'_persistence_day{day_idx}.npz'
    if cache.exists() and not force:
        z = np.load(cache, allow_pickle=True)
        return {k: z[k] for k in ('pred_anchor', 'true_anchor',
                                  'pred_rock', 'true_rock')}

    data = load_data_for_day(day_idx)
    sti = data['sensor_type_indices']
    a_s, a_e = sti['锚杆']
    r_s, r_e = sti['围岩']

    last_input = data['test_X'][:, -1, :]               # (N, total_features)
    persist_a = last_input[:, a_s:a_e]
    persist_r = last_input[:, r_s:r_e]
    true_a    = data['test_y']['next']['锚杆']
    true_r    = data['test_y']['next']['围岩']

    cfg = make_config()
    ev  = ModelEvaluator(cfg, data['scalers'])
    out = {
        'pred_anchor': ev.inverse_transform(persist_a, '锚杆'),
        'true_anchor': ev.inverse_transform(true_a,    '锚杆'),
        'pred_rock':   ev.inverse_transform(persist_r, '围岩'),
        'true_rock':   ev.inverse_transform(true_r,    '围岩'),
    }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **out)
    return out


def _metrics(true: np.ndarray, pred: np.ndarray) -> dict:
    """MAE / RMSE / MAPE / R²;true/pred 同形状。"""
    err = pred - true
    mae = float(np.abs(err).mean())
    rmse = float(np.sqrt((err ** 2).mean()))
    mask = np.abs(true) > 1e-8
    mape = float(np.mean(np.abs(err[mask] / true[mask])) * 100) if mask.any() else np.nan
    var = float(((true - true.mean()) ** 2).mean())
    r2  = float(1.0 - (err ** 2).mean() / var) if var > 0 else np.nan
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'R2': r2}


def collect_skill_records(days: list[int], models: list[str]) -> pd.DataFrame:
    """对 (model, day, sensor_type) 计算: 模型/持续基线 各自指标 + skill score
    + 动态段(top-20% by |true-persist|) 指标。"""
    rows = []
    for d in days:
        try:
            persist = get_persistence(d)
        except Exception as ex:
            print(f'  跳过 day{d} persistence: {ex}')
            continue

        for m in models:
            if m not in EXPERIMENTS:
                continue
            ck = _ckpt_path(m, d)
            if not ck.exists():
                continue
            try:
                p = get_predictions(m, d)
            except Exception as ex:
                print(f'  跳过 {m} day{d}: {ex}')
                continue

            for key, st in [('anchor', '锚杆'), ('rock', '围岩')]:
                true = p[f'true_{key}']
                pred = p[f'pred_{key}']
                pers = persist[f'pred_{key}']

                m_full = _metrics(true, pred)
                m_pers = _metrics(true, pers)

                # Skill score: 1 - model/persist (>0 即超过朴素基线)
                sk_mae  = 1.0 - m_full['MAE']  / m_pers['MAE']  if m_pers['MAE']  > 0 else np.nan
                sk_rmse = 1.0 - m_full['RMSE'] / m_pers['RMSE'] if m_pers['RMSE'] > 0 else np.nan
                sk_mse  = 1.0 - (m_full['RMSE'] ** 2) / (m_pers['RMSE'] ** 2) if m_pers['RMSE'] > 0 else np.nan

                # 动态段: 信号真在变的 top 20% 样本.
                # Bug fix: 准离散信号里大量 cell 的 change=0, 直接 percentile(80)
                # 会塌缩成 0, 让 mask = 全集. 改为先剔除 change=0 再取百分位。
                change = np.abs(true - pers)
                eps = max(1e-8, 1e-4 * float(np.abs(true).mean() or 1.0))
                nonzero = change[change > eps]
                if nonzero.size >= 10:
                    thr = float(np.percentile(nonzero, 80))
                    mask = change >= thr
                    if mask.sum() >= 10:
                        m_dyn      = _metrics(true[mask], pred[mask])
                        m_dyn_pers = _metrics(true[mask], pers[mask])
                        sk_dyn_mae = 1.0 - m_dyn['MAE']  / m_dyn_pers['MAE']  if m_dyn_pers['MAE']  > 0 else np.nan
                        sk_dyn_mse = 1.0 - (m_dyn['RMSE'] ** 2) / (m_dyn_pers['RMSE'] ** 2) if m_dyn_pers['RMSE'] > 0 else np.nan
                    else:
                        m_dyn = m_dyn_pers = {k: np.nan for k in ['MAE', 'RMSE', 'MAPE', 'R2']}
                        sk_dyn_mae = sk_dyn_mse = np.nan
                else:
                    m_dyn = m_dyn_pers = {k: np.nan for k in ['MAE', 'RMSE', 'MAPE', 'R2']}
                    sk_dyn_mae = sk_dyn_mse = np.nan

                rows.append({
                    'model': m, 'day': d, 'sensor': key,
                    'MAE': m_full['MAE'], 'RMSE': m_full['RMSE'],
                    'MAE_pers': m_pers['MAE'], 'RMSE_pers': m_pers['RMSE'],
                    'Skill_MAE': sk_mae, 'Skill_MSE': sk_mse,
                    'MAE_dyn': m_dyn['MAE'], 'RMSE_dyn': m_dyn['RMSE'],
                    'MAE_dyn_pers': m_dyn_pers['MAE'], 'RMSE_dyn_pers': m_dyn_pers['RMSE'],
                    'Skill_MAE_dyn': sk_dyn_mae, 'Skill_MSE_dyn': sk_dyn_mse,
                })
    return pd.DataFrame(rows)


def gen_skill_tables(days: list[int]):
    """生成 skill score 表 + 动态段指标表 (7天平均)."""
    all_models = list(EXPERIMENTS.keys())
    df = collect_skill_records(days, all_models)
    if df.empty:
        print('  collect_skill_records 返回空, 跳过 skill 表')
        return

    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    # 7 天平均
    avg = (df.groupby(['model', 'sensor']).mean(numeric_only=True)
             .reset_index()
             .drop(columns=['day'], errors='ignore'))

    def _pivot(df, value_cols, name_map):
        wide = df.pivot(index='model', columns='sensor', values=value_cols)
        wide.columns = [f'{name_map.get(s, s)}_{c}' for c, s in wide.columns]
        return wide

    name_map = {'anchor': 'Anchor', 'rock': 'Rock'}

    # ① Skill score 表 (相对持续基线的提升)
    skill_cols = ['Skill_MAE', 'Skill_MSE', 'Skill_MAE_dyn', 'Skill_MSE_dyn']
    skill_tbl = _pivot(avg, skill_cols, name_map)
    # 规范列顺序
    order = []
    for st in ['Anchor', 'Rock']:
        for c in skill_cols:
            order.append(f'{st}_{c}')
    skill_tbl = skill_tbl[[c for c in order if c in skill_tbl.columns]]

    # ② 动态段 MAE/RMSE 表
    dyn_cols = ['MAE_dyn', 'RMSE_dyn', 'MAE_dyn_pers', 'RMSE_dyn_pers']
    dyn_tbl = _pivot(avg, dyn_cols, name_map)
    order = [f'{st}_{c}' for st in ['Anchor', 'Rock'] for c in dyn_cols]
    dyn_tbl = dyn_tbl[[c for c in order if c in dyn_tbl.columns]]

    # ③ 持续基线本身的指标(给读者一个参考)
    pers_cols = ['MAE_pers', 'RMSE_pers']
    pers_tbl = _pivot(avg[avg['model'] == avg['model'].iloc[0]], pers_cols, name_map)
    pers_tbl.index = ['Persistence']

    # 输出 - 分组 (baseline / ablation), 改用 display name
    def _slice(tbl, group):
        idx = [g for g in group if g in tbl.index]
        sub = tbl.loc[idx].copy()
        sub.index = [DISPLAY[g] for g in idx]
        sub.index.name = 'Model'
        return sub.round(4)

    skill_base = _slice(skill_tbl, BASELINE_GROUP)
    skill_abl  = _slice(skill_tbl, ABLATION_GROUP)
    dyn_base   = _slice(dyn_tbl,   BASELINE_GROUP)
    dyn_abl    = _slice(dyn_tbl,   ABLATION_GROUP)

    skill_base.to_excel(TABLE_DIR / 'table_skill_baseline.xlsx')
    skill_abl.to_excel( TABLE_DIR / 'table_skill_ablation.xlsx')
    dyn_base.to_excel(  TABLE_DIR / 'table_dynamic_baseline.xlsx')
    dyn_abl.to_excel(   TABLE_DIR / 'table_dynamic_ablation.xlsx')

    # 同时存 csv 与逐天明细
    skill_base.to_csv(TABLE_DIR / 'table_skill_baseline.csv')
    skill_abl.to_csv( TABLE_DIR / 'table_skill_ablation.csv')
    dyn_base.to_csv(  TABLE_DIR / 'table_dynamic_baseline.csv')
    dyn_abl.to_csv(   TABLE_DIR / 'table_dynamic_ablation.csv')
    df.to_excel(TABLE_DIR / 'table_skill_perday_detail.xlsx', index=False)

    # markdown 索引 (手写, 不依赖 tabulate)
    def _df_to_md(df: pd.DataFrame, higher_better_cols=None) -> str:
        higher_better_cols = higher_better_cols or set()
        cols = df.columns.tolist()
        lines = ['| Model | ' + ' | '.join(cols) + ' |',
                 '| :--- ' + '| ---: ' * len(cols) + '|']
        bests = {}
        for c in cols:
            series = df[c]
            if series.notna().any():
                bests[c] = series.idxmax() if c in higher_better_cols else series.idxmin()
        for idx, row in df.iterrows():
            cells = []
            for c in cols:
                v = row[c]
                txt = '-' if pd.isna(v) else f'{v:.4f}'
                if idx == bests.get(c) and not pd.isna(v):
                    txt = f'**{txt}**'
                cells.append(txt)
            lines.append('| ' + str(idx) + ' | ' + ' | '.join(cells) + ' |')
        return '\n'.join(lines)

    skill_higher = {c for c in skill_base.columns}  # skill 越大越好

    md_parts = []
    md_parts.append('### Persistence baseline (naive y[t+1]=y[t]) — 平均参考值')
    md_parts.append(_df_to_md(pers_tbl.round(4)))
    md_parts.append('\n### Table 5-3  Skill Score (1 − model/persist;> 0 即超过朴素基线;'
                    + ('7 天' if len(days) >= 7 else f'{len(days)} 天') + '平均)')
    md_parts.append('\n#### Baseline 对比')
    md_parts.append(_df_to_md(skill_base, higher_better_cols=skill_higher))
    md_parts.append('\n#### Ablation 对比')
    md_parts.append(_df_to_md(skill_abl, higher_better_cols=skill_higher))
    md_parts.append('\n### Table 5-4  动态段(|y-y_persist| top-20% 样本)指标')
    md_parts.append('\n#### Baseline 对比')
    md_parts.append(_df_to_md(dyn_base))
    md_parts.append('\n#### Ablation 对比')
    md_parts.append(_df_to_md(dyn_abl))
    (TABLE_DIR / 'tables_skill.md').write_text('\n'.join(md_parts), encoding='utf-8')

    print(f'  -> Skill / dynamic 表已写入 {TABLE_DIR}')

    # ④ R²_pers 单独表 (= Skill_MSE,数学等价但论文里叫 R² 审稿人更熟)
    r2_cols = ['Skill_MSE', 'Skill_MSE_dyn']
    r2_tbl = _pivot(avg, r2_cols, name_map)
    r2_tbl.columns = [
        c.replace('Skill_MSE_dyn', 'R2_pers_dyn').replace('Skill_MSE', 'R2_pers')
        for c in r2_tbl.columns
    ]
    r2_order = [f'{st}_{c}' for st in ['Anchor', 'Rock']
                              for c in ['R2_pers', 'R2_pers_dyn']]
    r2_tbl = r2_tbl[[c for c in r2_order if c in r2_tbl.columns]]
    r2_base = _slice(r2_tbl, BASELINE_GROUP)
    r2_abl  = _slice(r2_tbl, ABLATION_GROUP)
    r2_base.to_excel(TABLE_DIR / 'table_r2_pers_baseline.xlsx')
    r2_abl.to_excel( TABLE_DIR / 'table_r2_pers_ablation.xlsx')
    r2_base.to_csv(  TABLE_DIR / 'table_r2_pers_baseline.csv')
    r2_abl.to_csv(   TABLE_DIR / 'table_r2_pers_ablation.csv')

    # markdown 单独追加 R²_pers 表 (Table 5-3 candidate)
    r2_higher = set(r2_base.columns)
    r2_md = []
    r2_md.append('### Table 5-3 (alt)  Persistence-relative R²  '
                 r'($R^2_{\mathrm{pers}} = 1 - \mathrm{MSE}_{\mathrm{model}} / '
                 r'\mathrm{MSE}_{\mathrm{persistence}}$;'
                 + ('7 天' if len(days) >= 7 else f'{len(days)} 天') + '平均)')
    r2_md.append('\n#### Baseline 对比')
    r2_md.append(_df_to_md(r2_base, higher_better_cols=r2_higher))
    r2_md.append('\n#### Ablation 对比')
    r2_md.append(_df_to_md(r2_abl,  higher_better_cols=r2_higher))
    (TABLE_DIR / 'tables_r2_pers.md').write_text('\n'.join(r2_md), encoding='utf-8')

    # 用 MAE_model / MAE_pers 这个相对比值替代 Skill Score 出柱图,
    # 因为 Skill = 1 - 比值, 比值远大于 1 时 Skill 是 -90 之类难看的数字。
    # 比值: <1 模型胜, =1 平手, >1 模型输给持续基线 (用红色标记)。
    fig_relative_bars(avg, BASELINE_GROUP, 'baseline')
    fig_relative_bars(avg, ABLATION_GROUP, 'ablation')

    # 同一份 avg 数据再出 R²_pers 柱图 (不需要重算 → 与 Skill_MSE 同源)
    fig_r2_pers_bars(avg, BASELINE_GROUP, 'baseline')
    fig_r2_pers_bars(avg, ABLATION_GROUP, 'ablation')


def fig_relative_bars(avg: pd.DataFrame, group: list[str], tag: str):
    sub = avg[avg['model'].isin(group)].copy()
    if sub.empty:
        return
    sub['Anchor_ratio_MAE'] = sub.where(sub['sensor'] == 'anchor')['MAE'] / sub.where(sub['sensor'] == 'anchor')['MAE_pers']
    sub['Rock_ratio_MAE'] = sub.where(sub['sensor'] == 'rock')['MAE'] / sub.where(sub['sensor'] == 'rock')['MAE_pers']
    sub['Anchor_ratio_MAE_dyn'] = sub.where(sub['sensor'] == 'anchor')['MAE_dyn'] / sub.where(sub['sensor'] == 'anchor')['MAE_dyn_pers']
    sub['Rock_ratio_MAE_dyn'] = sub.where(sub['sensor'] == 'rock')['MAE_dyn'] / sub.where(sub['sensor'] == 'rock')['MAE_dyn_pers']
    by_model = sub.groupby('model').first()
    panels = [
        ('Anchor — All samples', by_model['Anchor_ratio_MAE'].reindex(group)),
        ('Anchor — Dynamic top 20%', by_model['Anchor_ratio_MAE_dyn'].reindex(group)),
        ('Rock — All samples', by_model['Rock_ratio_MAE'].reindex(group)),
        ('Rock — Dynamic top 20%', by_model['Rock_ratio_MAE_dyn'].reindex(group)),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.2))
    for ax, (title, vals) in zip(axes.flat, panels):
        names = [DISPLAY[g] for g in vals.index]
        x = np.arange(len(names))
        colors = ['#C0392B' if (not pd.isna(v) and v > 1) else PALETTE[i % len(PALETTE)]
                  for i, v in enumerate(vals.values)]
        ax.bar(x, vals.values, color=colors, width=0.62, zorder=2)
        ax.axhline(1.0, color='black', linewidth=1, linestyle='--', label='Persistence baseline', zorder=1)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=28, ha='right')
        ax.set_ylabel('MAE / MAE_persistence')
        ax.set_title(title, fontsize=10)
        annotate_bar_values(ax, x, vals.values, fontsize=8)
        if 'Rock' in title:
            ax.set_yscale('log')
        ax.grid(axis='y', linestyle=':', alpha=0.35)
    finalize_bar_grid(fig, axes)
    _save(fig, f'fig_relative_mae_{tag}')


def fig_r2_pers_bars(avg: pd.DataFrame, group: list[str], tag: str):
    sub = avg[avg['model'].isin(group)].copy()
    if sub.empty:
        return
    sub['Anchor_R2pers'] = sub.where(sub['sensor'] == 'anchor')['Skill_MSE']
    sub['Anchor_R2pers_dyn'] = sub.where(sub['sensor'] == 'anchor')['Skill_MSE_dyn']
    sub['Rock_R2pers'] = sub.where(sub['sensor'] == 'rock')['Skill_MSE']
    sub['Rock_R2pers_dyn'] = sub.where(sub['sensor'] == 'rock')['Skill_MSE_dyn']
    by_model = sub.groupby('model').first()
    panels = [
        ('Anchor — All samples', by_model['Anchor_R2pers'].reindex(group)),
        ('Anchor — Dynamic top 20%', by_model['Anchor_R2pers_dyn'].reindex(group)),
        ('Rock — All samples', by_model['Rock_R2pers'].reindex(group)),
        ('Rock — Dynamic top 20%', by_model['Rock_R2pers_dyn'].reindex(group)),
    ]
    highlight, normal = '#C0392B', '#3F6FB6'
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.2))
    for ax, (title, vals) in zip(axes.flat, panels):
        names = [DISPLAY[g] for g in vals.index]
        x = np.arange(len(names))
        colors = [highlight if g == 'Full_Model' else normal for g in vals.index]
        ax.bar(x, vals.values, color=colors, width=0.62, zorder=2)
        ax.axhline(0.0, color='black', linewidth=1, linestyle='--', label='Persistence (R² = 0)', zorder=1)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=28, ha='right')
        ax.set_ylabel(r'$R^2_{\mathrm{pers}}$')
        ax.set_title(title, fontsize=10)
        for xi, v in zip(x, vals.values):
            if not pd.isna(v):
                ax.text(xi, v, f'{v:.2f}', ha='center', va='bottom' if v >= 0 else 'top', fontsize=8)
        ax.grid(axis='y', linestyle=':', alpha=0.45)
    finalize_bar_grid(fig, axes)
    _save(fig, f'fig_r2_pers_{tag}')


# ── 复杂区检测 + 多模型对比 ───────────────────────────────────────────────────

def detect_regions(signal: np.ndarray, win: int = 200) -> dict:
    """返回 3 个区段 (start, end), 尽量互不重叠 (>= win 距离).

    - Peak region:        smoothed 信号偏离均值最大处 (持续高/低)
    - Transition region:  smoothed 信号梯度最大处 (single edge / step change)
    - Oscillation region: 状态翻转最频繁处 (用阈值离散化后, 滚动求翻转次数)
    """
    n = len(signal)
    half = win // 2
    s = pd.Series(signal).astype(float)

    # 多档平滑
    smooth_heavy  = s.rolling(window=max(40, win // 5),  center=True,
                              min_periods=1).mean().values  # 用于检测持续变化
    smooth_light  = s.rolling(window=max(10, win // 20), center=True,
                              min_periods=1).mean().values

    # ① 峰值: |smooth - mean| 最大
    centered = np.abs(smooth_heavy - smooth_heavy.mean())

    # ② 平滑后的梯度 (找 step 边缘)
    grad = pd.Series(np.abs(np.gradient(smooth_heavy))).rolling(
        max(20, win // 10), center=True, min_periods=1).mean().values

    # ③ 状态翻转次数: 把信号离散到中位数上下两态, 数 ±1 翻转
    states = (signal > np.median(signal)).astype(int)
    flips = np.zeros(n)
    flips[1:] = np.abs(np.diff(states))
    osc = pd.Series(flips).rolling(window=win, center=True,
                                   min_periods=1).sum().values

    def _pick_unique(score, taken, min_dist):
        for i in np.argsort(-score):
            if half <= i <= n - half - 1 and \
               all(abs(int(i) - t) >= min_dist for t in taken):
                return int(i)
        return int(max(half, min(n - half - 1, int(np.argmax(score)))))

    taken = []
    p = _pick_unique(centered, taken, win); taken.append(p)
    t = _pick_unique(grad,     taken, win); taken.append(t)
    o = _pick_unique(osc,      taken, win); taken.append(o)

    return {
        'Peak region':         (p - half, p + half),
        'Transition region':   (t - half, t + half),
        'Oscillation region':  (o - half, o + half),
    }


def fig_complex_regions(day_idx: int,
                        group: list[str], tag: str,
                        win: int = 200, smooth: int = 15):
    """每个传感器类型自动选一只方差最大的传感器, 在它的 3 个复杂区上叠加多模型曲线.

    显示策略 (避免 5 条曲线 + 高频噪声混成色块):
      - 测量值原始: 浅灰散点
      - 测量值平滑: 粗黑线
      - 每个模型: 各自平滑后的预测线
    """
    valid = [m for m in group if m in EXPERIMENTS and _ckpt_path(m, day_idx).exists()]
    if not valid:
        print(f'  fig_complex_regions: 无模型可用, 跳过 ({tag})')
        return
    persist = get_persistence(day_idx)
    preds = {m: get_predictions(m, day_idx) for m in valid}
    ref = preds[valid[0]]

    a_idx = int(np.argmax([ref['true_anchor'][:, i].std()
                           for i in range(ref['true_anchor'].shape[1])]))
    r_idx = int(np.argmax([ref['true_rock'][:, i].std()
                           for i in range(ref['true_rock'].shape[1])]))
    sensor_picks = [('anchor', a_idx, f'Anchor #{a_idx + 1}'),
                    ('rock',   r_idx, f'Rock mass #{r_idx + 1}')]

    def _smooth(a, k):
        if k <= 1:
            return a
        return pd.Series(a).rolling(window=k, center=True,
                                    min_periods=1).mean().values

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 7.6))
    legend_handles = None
    for r, (kind, idx, lbl) in enumerate(sensor_picks):
        true = ref[f'true_{kind}'][:, idx]
        pers = persist[f'pred_{kind}'][:, idx]
        regions = detect_regions(true, win=win)

        for c, (rname, (s, e)) in enumerate(regions.items()):
            ax = axes[r, c]
            x = np.arange(s, e)
            t_seg = true[s:e]
            p_seg = pers[s:e]
            t_smooth = _smooth(t_seg, smooth)
            p_smooth = _smooth(p_seg, smooth)

            # 原始测量散点 (灰)
            ax.scatter(x, t_seg, s=4, color='lightgray', alpha=0.45,
                       label='Measured (raw)', zorder=1)
            ax.plot(x, t_smooth, color='black', linewidth=1.7,
                    label='Measured (smoothed)', zorder=10)
            ax.plot(x, p_smooth, color='#999999', linewidth=1.0,
                    linestyle=':', label='Persistence', alpha=0.85, zorder=4)
            # 模型预测本身已经较平滑, 不再做平均, 直接画原始预测线
            for i, m in enumerate(valid):
                pr = preds[m][f'pred_{kind}'][:, idx][s:e]
                ax.plot(x, pr, color=PALETTE[i % len(PALETTE)],
                        linewidth=1.3, linestyle='--', alpha=0.95,
                        label=DISPLAY[m], zorder=5 + i)

            # 自适应 y 轴 (基于 measured 真实范围 + 一点 margin)
            lo, hi = float(t_seg.min()), float(t_seg.max())
            rng = hi - lo
            if rng < 1e-3 * max(abs(hi), 1.0):
                mid = 0.5 * (lo + hi)
                half = max(0.01 * abs(mid), 1e-3)
                lo, hi = mid - half, mid + half
                rng = hi - lo
            ax.set_ylim(lo - 0.2 * rng, hi + 0.25 * rng)
            ax.set_title(f'({chr(97 + r * 3 + c)}) {lbl} — {rname}',
                         fontsize=10)
            ax.set_xlabel('Test sample index')
            if c == 0:
                ax.set_ylabel('Value')
            if legend_handles is None and r == 0 and c == 0:
                legend_handles = (ax.get_legend_handles_labels())

    # 整图统一图例 (放在底部, 避免遮数据)
    if legend_handles is not None:
        fig.legend(*legend_handles, ncol=min(8, 3 + len(valid)),
                   loc='lower center', bbox_to_anchor=(0.5, 0.02),
                   frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 0.98))
    _save(fig, f'fig_complex_regions_{tag}_day{day_idx}')


# ── 数据汇总辅助 ──────────────────────────────────────────────────────────────

def load_summary() -> pd.DataFrame:
    if not SUMMARY_XLS.exists():
        raise FileNotFoundError(f'未找到 {SUMMARY_XLS}, 请先跑 experiment_main.py')
    return pd.read_excel(SUMMARY_XLS, sheet_name='平均结果(论文对比)')


def best_in_col(series: pd.Series, metric: str):
    return series.idxmax() if metric == 'R2' else series.idxmin()


# ── 表 ────────────────────────────────────────────────────────────────────────

def _format_table(df_avg: pd.DataFrame, group: list[str]) -> pd.DataFrame:
    # 仅保留 summary 中存在的模型 (允许部分实验)
    available = [g for g in group if g in df_avg['模型'].values]
    if not available:
        return pd.DataFrame()
    sub = df_avg[df_avg['模型'].isin(available)].set_index('模型').loc[available].copy()
    sub.index = [DISPLAY[m] for m in sub.index]
    sub.index.name = 'Model'
    sub = sub.rename(columns={
        '锚杆_MAE': 'Anchor MAE', '锚杆_RMSE': 'Anchor RMSE',
        '锚杆_MAPE': 'Anchor MAPE(%)', '锚杆_R²': 'Anchor R²',
        '围岩_MAE': 'Rock MAE', '围岩_RMSE': 'Rock RMSE',
        '围岩_MAPE': 'Rock MAPE(%)', '围岩_R²': 'Rock R²',
    })
    return sub.round(4)


def _md_table(df: pd.DataFrame, title: str) -> str:
    """DataFrame 转 markdown, 每列粗体最优值."""
    cols = df.columns.tolist()
    is_higher_better = {c: ('R²' in c) for c in cols}
    bests = {c: (df[c].idxmax() if is_higher_better[c] else df[c].idxmin()) for c in cols}

    lines = [f'### {title}', '',
             '| Model | ' + ' | '.join(cols) + ' |',
             '| :--- ' + '| :---: ' * len(cols) + '|']
    for idx, row in df.iterrows():
        cells = []
        for c in cols:
            val = row[c]
            txt = f'{val:.4f}' if not pd.isna(val) else '-'
            if idx == bests[c]:
                txt = f'**{txt}**'
            cells.append(txt)
        lines.append('| ' + idx + ' | ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines) + '\n'


def gen_tables():
    df_avg = load_summary()
    base = _format_table(df_avg, BASELINE_GROUP)
    abl  = _format_table(df_avg, ABLATION_GROUP)

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    md_parts = []
    if not base.empty:
        base.to_excel(TABLE_DIR / 'table_baseline_comparison.xlsx')
        base.to_csv(TABLE_DIR / 'table_baseline_comparison.csv')
        md_parts.append(_md_table(base, 'Table 5-1  Baseline comparison'))
    else:
        print('  [skip] baseline 表: summary 中没有 BASELINE_GROUP 任何模型')
    if not abl.empty:
        abl.to_excel(TABLE_DIR / 'table_ablation_comparison.xlsx')
        abl.to_csv(TABLE_DIR / 'table_ablation_comparison.csv')
        md_parts.append(_md_table(abl, 'Table 5-2  Ablation study'))
    else:
        print('  [skip] ablation 表: summary 中没有 ABLATION_GROUP 任何模型')

    if md_parts:
        (TABLE_DIR / 'tables.md').write_text('\n'.join(md_parts), encoding='utf-8')
        print(f'  -> 表已写入 {TABLE_DIR}')


# ── 图 1: 训练 / 验证 loss 曲线 (Fig 8 风格) ──────────────────────────────────

def _mse_display_scale(train: np.ndarray, val: np.ndarray) -> tuple[float, str]:
    """按数量级选纵轴缩放，接近常见论文「×10⁻³」写法。"""
    m = float(max(np.max(train), np.max(val), 1e-12))
    if m < 2e-4:
        return 1e5, r'MSE ($\times 10^{-5}$)'
    if m < 2e-3:
        return 1e4, r'MSE ($\times 10^{-4}$)'
    if m < 0.02:
        return 1e3, r'MSE ($\times 10^{-3}$)'
    return 1.0, 'MSE'


def fig_loss_curves(day_idx: int):
    """训练/验证 MSE 曲线，版式参照预期效果图 Fig.8。

    - 左: BiLSTM（与参考图 (a) 对应）; 右: Full_Model（完整模型，对应本文方法）
    - 黑色实线 Training / 红色实线 Validation，不做平滑以保留 epoch 间起伏
    - 数据优先取 loss_curve_runs 长训（见 get_history）
    """
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.5))
    # (实验名, 子图标题)
    targets = [
        ('BiLSTM', '(a) Bi-LSTM module training'),
        ('Full_Model', '(b) Full model training'),
    ]
    for ax, (exp, title) in zip(axes, targets):
        h = get_history(exp, day_idx)
        if h is None or not h.get('train_loss'):
            ax.set_title(f'{title}\n(no history)')
            ax.set_xlabel('Epoch')
            continue
        train = np.asarray(h['train_loss'], dtype=float)
        val = np.asarray(h['val_loss'], dtype=float)
        n = min(len(train), len(val))
        train, val = train[:n], val[:n]
        epochs = np.arange(1, n + 1)

        scale, ylabel = _mse_display_scale(train, val)
        ax.plot(epochs, train * scale, color='black', linewidth=1.2,
                label='Training loss')
        ax.plot(epochs, val * scale, color='#C0392B', linewidth=1.2,
                label='Validation loss')

        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11)
        ax.legend(frameon=False, fontsize=9, loc='upper right')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, linestyle=':', alpha=0.45)

    fig.tight_layout()
    _save(fig, 'fig_loss_curves')


# ── 图 2: 指标分组柱状图 (基线 + 消融) ─────────────────────────────────────────

def fig_metric_bars():
    df_avg = load_summary()

    def _draw(group, title, fname):
        avail = [g for g in group if g in df_avg['模型'].values]
        if len(avail) < 2:
            print(f'  [skip] {fname}: 仅 {len(avail)} 个模型可绘 (<2)')
            return
        sub = df_avg[df_avg['模型'].isin(avail)].set_index('模型').loc[avail]
        names = [DISPLAY[m] for m in avail]
        group = avail  # downstream best-index calc 用 avail
        x = np.arange(len(names))
        w = 0.36

        fig, axes = plt.subplots(2, 4, figsize=(14, 6.4), sharex=True)
        for r, st in enumerate(SENSOR_TYPES):
            for c, m in enumerate(METRICS):
                ax = axes[r, c]
                col = f'{st}_{"R²" if m == "R2" else m}'
                vals = sub[col].values
                colors = [PALETTE[i % len(PALETTE)] for i in range(len(group))]
                # 高亮最佳
                best = np.argmax(vals) if m == 'R2' else np.argmin(vals)
                edge = ['black' if i == best else 'none' for i in range(len(group))]
                lw   = [1.6 if i == best else 0 for i in range(len(group))]
                ax.bar(x, vals, width=0.62, color=colors,
                       edgecolor=edge, linewidth=lw)
                ax.set_title(f'{SENSOR_DISPLAY[st]} - {METRIC_LABELS[m]}')
                ax.set_xticks(x)
                ax.set_xticklabels(names, rotation=25, ha='right')
                if m == 'R2':
                    span = vals.max() - vals.min()
                    ax.set_ylim(vals.min() - span * 0.4, vals.max() + span * 0.2)
                else:
                    ax.set_ylim(0, vals.max() * 1.18)
                for xi, v in zip(x, vals):
                    ax.text(xi, v, f'{v:.4f}' if m != 'MAPE' else f'{v:.2f}',
                            ha='center', va='bottom', fontsize=8)
        fig.suptitle(title, y=1.02, fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        _save(fig, fname)

    _draw(BASELINE_GROUP, 'Baseline comparison on test set (averaged over 7 days)',
          'fig_metrics_baseline')
    _draw(ABLATION_GROUP, 'Ablation study on test set (averaged over 7 days)',
          'fig_metrics_ablation')


# ── 图 3: 单模型预测 vs 真实 (Fig 9 风格) ─────────────────────────────────────

def fig_pred_vs_real(day_idx: int, exp_name: str = 'Full_Model',
                     n_show: int = 6, window: int = 2000,
                     smooth: int = 60):
    """6 个传感器, 测试集前 ``window`` 个连续点的 measured vs predicted.
    measured 加滑窗均值后画线 (避免 5s 间隔的高频噪声看成黑色色块)."""
    p = get_predictions(exp_name, day_idx)
    pred = np.concatenate([p['pred_anchor'], p['pred_rock']], axis=1)
    true = np.concatenate([p['true_anchor'], p['true_rock']], axis=1)

    n_anchor = p['pred_anchor'].shape[1]
    n_rock   = p['pred_rock'].shape[1]
    labels = [f'Anchor #{i+1}'   for i in range(n_anchor)] + \
             [f'Rock mass #{i+1}' for i in range(n_rock)]
    n_show = min(n_show, pred.shape[1])

    n_step = pred.shape[0]
    end = min(window, n_step)
    x = np.arange(end)

    def _smooth(a, k):
        if k <= 1:
            return a
        return pd.Series(a).rolling(window=k, center=True,
                                    min_periods=1).mean().values

    rows, cols = 2, 3
    fig, axes = plt.subplots(rows, cols, figsize=(13, 6.6), sharex=True)
    axes = axes.flatten()

    for k in range(n_show):
        ax = axes[k]
        t_raw = true[:end, k]
        p_raw = pred[:end, k]
        true_s = _smooth(t_raw, smooth)
        pred_s = _smooth(p_raw, smooth)
        # 原始测量值用浅灰散点表示 (体现高频噪声)
        ax.scatter(x, t_raw, s=2, color='lightgray', alpha=0.4,
                   label='Measured (raw)')
        ax.plot(x, true_s, color='black', linewidth=1.4,
                label='Measured (smoothed)')
        ax.plot(x, pred_s, color='#1f77b4', linewidth=1.4,
                linestyle='--', label='Predicted')
        # y 轴根据真实+预测的中段数据自动收紧
        all_vals = np.concatenate([t_raw, p_raw])
        lo, hi = np.percentile(all_vals, 2), np.percentile(all_vals, 98)
        rng = hi - lo
        if rng < 1e-3 * max(abs(hi), 1.0):
            # 数据近乎不变, 给一个相对的范围
            mid = 0.5 * (lo + hi)
            half = max(0.005 * abs(mid), 1e-4)
            lo, hi = mid - half, mid + half
        pad = max(rng * 0.30, 1e-4)
        ax.set_ylim(lo - pad, hi + pad)
        ax.text(0.97, 0.05, labels[k], transform=ax.transAxes,
                ha='right', va='bottom', fontsize=9,
                bbox=dict(facecolor='white', edgecolor='gray',
                          boxstyle='round,pad=0.25', alpha=0.85))
        if k == 0:
            ax.legend(frameon=False, loc='best', fontsize=7)
        if k >= cols:
            ax.set_xlabel('Test sample index')
        if k % cols == 0:
            ax.set_ylabel('Value')

    for k in range(n_show, len(axes)):
        axes[k].axis('off')

    fig.tight_layout()
    _save(fig, f'fig_pred_vs_real_{exp_name.replace("/", "_")}_day{day_idx}')


# ── 图 4: 多模型对比时序 + 散点 (Fig 10 风格) ────────────────────────────────

def fig_compare_timeline(day_idx: int,
                         models: tuple[str, ...] = ('Full_Model', 'BiLSTM', 'CNN-LSTM', 'Transformer'),
                         sensor_kind: str = 'auto', sensor_idx: int = -1):
    valid_models = [m for m in models if m in EXPERIMENTS and _ckpt_path(m, day_idx).exists()]
    if not valid_models:
        print('  fig_compare_timeline: 无可用模型, 跳过')
        return
    ref = get_predictions(valid_models[0], day_idx)
    if sensor_kind == 'auto' or sensor_idx < 0:
        cand = [('anchor', i, ref['true_anchor'][:, i].std()) for i in range(ref['true_anchor'].shape[1])]
        cand += [('rock', i, ref['true_rock'][:, i].std()) for i in range(ref['true_rock'].shape[1])]
        sensor_kind, sensor_idx, _ = max(cand, key=lambda t: t[2])
    true = ref['true_anchor'][:, sensor_idx] if sensor_kind == 'anchor' else ref['true_rock'][:, sensor_idx]
    series = {}
    for m in valid_models:
        d = get_predictions(m, day_idx)
        series[m] = d['pred_anchor'][:, sensor_idx] if sensor_kind == 'anchor' else d['pred_rock'][:, sensor_idx]
    sub = np.arange(len(true))
    def _smooth(a, k=60):
        return pd.Series(a).rolling(window=k, center=True, min_periods=1).mean().values
    n_models = len(series)
    fig = plt.figure(figsize=(13.5, 8.0))
    gs = fig.add_gridspec(
        2, n_models, height_ratios=[1.05, 0.85],
        hspace=0.42, wspace=0.28,
        top=0.92, bottom=0.12, left=0.06, right=0.98,
    )
    ax_top = fig.add_subplot(gs[0, :])
    ax_top.plot(sub, _smooth(true), color='black', linewidth=1.6, label='Measured')
    for i, (m, y) in enumerate(series.items()):
        ax_top.plot(sub, _smooth(y), color=PALETTE[i % len(PALETTE)],
                    linewidth=1.2, alpha=0.85, label=DISPLAY[m])
    loc_label = f'Anchor #{sensor_idx + 1}' if sensor_kind == 'anchor' else f'Rock mass #{sensor_idx + 1}'
    ax_top.set_xlabel('Test sample index')
    ax_top.set_ylabel('Value')
    ax_top.margins(x=0.01)
    ylo, yhi = ax_top.get_ylim()
    span = yhi - ylo
    ax_top.set_ylim(ylo - span * 0.02, yhi + span * 0.10)

    handles, labels = ax_top.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.01),
               ncol=min(6, len(handles)), frameon=False, fontsize=9)
    fig.text(0.5, 0.98, f'Time series — {loc_label}',
             ha='center', va='top', fontsize=10, transform=fig.transFigure)

    lo = float(min(true.min(), min(y.min() for y in series.values())))
    hi = float(max(true.max(), max(y.max() for y in series.values())))
    for i, (m, y) in enumerate(series.items()):
        ax = fig.add_subplot(gs[1, i])
        r2 = 1.0 - np.sum((y - true) ** 2) / np.sum((true - true.mean()) ** 2)
        ax.scatter(true, y, s=8, alpha=0.45, color=PALETTE[i % len(PALETTE)], edgecolor='none')
        ax.plot([lo, hi], [lo, hi], color='black', linewidth=1, linestyle='--')
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.margins(0)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(f'{DISPLAY[m]}  R² = {r2:.4f}', fontsize=10)
        ax.set_xlabel('Measured')
        if i == 0:
            ax.set_ylabel('Predicted')
    _save(fig, f'fig_compare_timeline_{sensor_kind}{sensor_idx + 1}_day{day_idx}')


# ── 图 5: 误差密度 + 箱线 (Fig 12 风格) ───────────────────────────────────────

def fig_error_density_box(day_idx: int, group: list[str], tag: str):
    errs_anchor, errs_rock = {}, {}
    for m in group:
        try:
            d = get_predictions(m, day_idx)
            errs_anchor[m] = (d['pred_anchor'] - d['true_anchor']).ravel()
            errs_rock[m]   = (d['pred_rock']   - d['true_rock']).ravel()
        except Exception as ex:
            print(f'  跳过 {m}: {ex}')

    if not errs_anchor:
        return

    try:
        from scipy.stats import gaussian_kde
        have_kde = True
    except Exception:
        have_kde = False

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.4))

    for r, (errs, st_label) in enumerate([(errs_anchor, 'Anchor'),
                                          (errs_rock,   'Rock mass')]):
        abs_p99 = max(np.percentile(np.abs(e), 99) for e in errs.values())
        x_lim = abs_p99 * 1.2 if abs_p99 > 0 else 1e-3

        # (a/c) density
        ax = axes[r, 0]
        xs = np.linspace(-x_lim, x_lim, 400)
        for i, (m, e) in enumerate(errs.items()):
            if have_kde:
                kde = gaussian_kde(e, bw_method=0.25)
                ax.plot(xs, kde(xs), color=PALETTE[i % len(PALETTE)],
                        linewidth=1.6, label=DISPLAY[m])
            else:
                ax.hist(e, bins=80, density=True, histtype='step',
                        color=PALETTE[i % len(PALETTE)],
                        linewidth=1.4, label=DISPLAY[m])
        ax.set_xlabel('Error')
        ax.set_ylabel('Distribution density')
        ax.set_title(f'({chr(97 + 2 * r)}) {st_label} - error density')
        ax.set_xlim(-x_lim, x_lim)
        if r == 0:
            ax.legend(frameon=False, fontsize=8, loc='upper center',
                      bbox_to_anchor=(0.5, -0.12), ncol=min(5, len(errs)))

        # (b/d) box
        ax = axes[r, 1]
        data = list(errs.values())
        names = [DISPLAY[m] for m in errs.keys()]
        bp = ax.boxplot(data, vert=True, patch_artist=True, showfliers=True,
                        widths=0.55,
                        flierprops=dict(marker='.', markersize=2, alpha=0.4))
        for i, patch in enumerate(bp['boxes']):
            patch.set_facecolor(PALETTE[i % len(PALETTE)])
            patch.set_alpha(0.8)
        for med in bp['medians']:
            med.set_color('black')
        ax.set_xticklabels(names, rotation=20, ha='right')
        ax.set_ylabel('Error')
        ax.set_ylim(-x_lim, x_lim)
        ax.set_title(f'({chr(98 + 2 * r)}) {st_label} - error box plot')

    fig.tight_layout()
    _save(fig, f'fig_error_density_box_{tag}_day{day_idx}')


# ── 图 6: 消融雷达图 ──────────────────────────────────────────────────────────

def fig_ablation_radar():
    df_avg = load_summary()
    avail = [g for g in ABLATION_GROUP if g in df_avg['模型'].values]
    if len(avail) < 3:
        print(f'  [skip] fig_ablation_radar: 仅 {len(avail)} 个消融模型可用 (<3)')
        return
    sub = df_avg[df_avg['模型'].isin(avail)].set_index('模型').loc[avail]
    radar_group = avail

    metric_cols = []
    metric_axes = []
    # 反向化误差类指标 (越小越好) → 相对得分 (越大越好)
    for st in SENSOR_TYPES:
        for m in METRICS:
            metric_cols.append(f'{st}_{"R²" if m == "R2" else m}')
            metric_axes.append(f'{SENSOR_DISPLAY[st]}\n{METRIC_LABELS[m]}')

    raw = sub[metric_cols].values.astype(float)
    score = np.zeros_like(raw)
    for j, c in enumerate(metric_cols):
        col = raw[:, j]
        if 'R²' in c:
            # R² 越大越好 → 直接归一化
            denom = (col.max() - col.min()) or 1.0
            score[:, j] = (col - col.min()) / denom
        else:
            # 越小越好 → 反向归一化
            denom = (col.max() - col.min()) or 1.0
            score[:, j] = 1.0 - (col - col.min()) / denom

    angles = np.linspace(0, 2 * np.pi, len(metric_cols), endpoint=False).tolist()
    angles += angles[:1]
    score_close = np.concatenate([score, score[:, :1]], axis=1)

    fig, ax = plt.subplots(figsize=(7.5, 7.5),
                           subplot_kw=dict(polar=True))
    for i, name in enumerate(radar_group):
        ax.plot(angles, score_close[i], color=PALETTE[i % len(PALETTE)],
                linewidth=1.8, label=DISPLAY[name])
        ax.fill(angles, score_close[i], color=PALETTE[i % len(PALETTE)],
                alpha=0.10)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_axes, fontsize=9)
    ax.set_yticklabels([])
    ax.set_ylim(0, 1.05)
    ax.set_title('Ablation study (normalized scores; outer = better)',
                 y=1.05, fontsize=12)
    ax.legend(loc='upper right', bbox_to_anchor=(1.32, 1.1), frameon=False)
    fig.tight_layout()
    _save(fig, 'fig_ablation_radar')


# ── 图 7: 测试集分段误差区间 (Fig 11 风格) ────────────────────────────────────

def fig_error_intervals(day_idx: int,
                        models: tuple[str, ...] = ('Full_Model', 'BiLSTM', 'LSTM'),
                        sensor_kind: str = 'rock', n_seg: int = 12):
    """把测试集按时间均分 n_seg 段, 画每段的平均误差 + 80%/100% 区间."""
    panels = []
    for m in models:
        if m not in EXPERIMENTS or not _ckpt_path(m, day_idx).exists():
            print(f'  跳过 {m}: 无 ckpt')
            continue
        d = get_predictions(m, day_idx)
        if sensor_kind == 'anchor':
            err = d['pred_anchor'] - d['true_anchor']
        else:
            err = d['pred_rock'] - d['true_rock']
        err = err.mean(axis=1)  # 每个时间步上对所有传感器求均误
        panels.append((m, err))

    if not panels:
        return

    n = len(panels)
    fig, axes = plt.subplots(n, 1, figsize=(8.5, 2.6 * n + 0.3),
                             sharex=True)
    if n == 1:
        axes = [axes]

    # 全局阈值: 取所有误差 95 百分位绝对值 → 各子图统一参考线
    abs_p95 = max(np.percentile(np.abs(e), 95) for _, e in panels)
    threshold = abs_p95 * 1.5
    ymax = abs_p95 * 2.5

    for ax, (m, err) in zip(axes, panels):
        seg = np.array_split(err, n_seg)
        seg_mean = np.array([s.mean() for s in seg])
        seg_p10  = np.array([np.percentile(s, 10) for s in seg])
        seg_p90  = np.array([np.percentile(s, 90) for s in seg])
        seg_min  = np.array([s.min() for s in seg])
        seg_max  = np.array([s.max() for s in seg])
        x = np.arange(1, n_seg + 1)
        ax.fill_between(x, seg_min, seg_max, color='#9DC3E6', alpha=0.25,
                        label='Interval of 100%')
        ax.fill_between(x, seg_p10, seg_p90, color='#5B9BD5', alpha=0.45,
                        label='Interval of 80%')
        ax.plot(x, seg_mean, color='#1F4E79', marker='o', linewidth=1.4,
                markersize=4, label='Average')
        ax.axhline(0, color='gray', linewidth=0.6)
        ax.axhline( threshold, color='#C0392B', linestyle='--', linewidth=0.9,
                    label=f'±{threshold:.3g} threshold' if ax is axes[0] else None)
        ax.axhline(-threshold, color='#C0392B', linestyle='--', linewidth=0.9)
        ax.set_ylim(-ymax, ymax)
        ax.set_ylabel('Error')
        ax.set_title(f'({DISPLAY[m]})', fontsize=10, loc='left')
        if ax is axes[0]:
            ax.legend(frameon=False, loc='upper right', fontsize=9, ncol=4)

    axes[-1].set_xlabel('Test segment')
    axes[-1].set_xticks(np.arange(1, n_seg + 1))
    fig.tight_layout()
    _save(fig, f'fig_error_intervals_{sensor_kind}_day{day_idx}')


# ── 工具 ──────────────────────────────────────────────────────────────────────

def _save(fig, name: str):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f'{name}.png', dpi=300, bbox_inches='tight')
    fig.savefig(FIG_DIR / f'{name}.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'    saved {FIG_DIR / name}.png/.pdf')


# ── 入口 ──────────────────────────────────────────────────────────────────────

ALL_TASKS = ['tables', 'loss', 'bars', 'pred_vs_real',
             'timeline', 'density', 'ablation_radar', 'error_band',
             'skill', 'complex']


def _run_one_variant(args, variant: str):
    """跑一遍指定 variant 的所有 figure/task。表写入 OUT_DIR/tables/, 图写入 OUT_DIR/figures/。"""
    setup_variant(variant)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    tasks = args.only or ALL_TASKS

    print('\n' + '=' * 70)
    print(f'[variant={variant}]  day={args.day}  seed={SEED}  '
          f'tasks={tasks}  ({len(BASELINE_GROUP)} 列基线)')
    print(f'输出: {OUT_DIR}')
    print('=' * 70)

    # 表 (无需模型预测, 仅读 SUMMARY)
    if 'tables' in tasks:
        print('\n[tables] 基线 / 消融 对比表...')
        gen_tables()

    # loss 曲线 (只读 history, ML 跳过)
    if 'loss' in tasks:
        print('\n[loss] 训练曲线 (DL only)...')
        fig_loss_curves(args.day)

    # 柱状图 (只用 summary)
    if 'bars' in tasks:
        print('\n[bars] 指标柱状图...')
        fig_metric_bars()

    if 'ablation_radar' in tasks:
        print('\n[ablation_radar] 消融雷达图...')
        fig_ablation_radar()

    needs_pred = {'pred_vs_real', 'timeline', 'density', 'error_band'} & set(tasks)
    if needs_pred and not args.no_predict:
        print('\n→ 接下来需要加载模型并产出预测...')

    if 'pred_vs_real' in tasks:
        print('\n[pred_vs_real] Full_Model 预测 vs 真实...')
        fig_pred_vs_real(args.day, 'Full_Model')

    if 'timeline' in tasks:
        print('\n[timeline] 多模型对比时序+散点...')
        # main variant 不含 Transformer, 退到 SVR; appendix 保留 Transformer
        tl_models = (('Full_Model', 'BiLSTM', 'CNN-LSTM', 'SVR')
                     if variant == 'main'
                     else ('Full_Model', 'BiLSTM', 'CNN-LSTM', 'Transformer', 'SVR'))
        fig_compare_timeline(args.day, models=tl_models,
                             sensor_kind='auto', sensor_idx=-1)

    if 'density' in tasks:
        print('\n[density] 误差密度+箱线 (基线 + 消融)...')
        fig_error_density_box(args.day, BASELINE_GROUP, 'baseline')
        fig_error_density_box(args.day, ABLATION_GROUP, 'ablation')

    if 'error_band' in tasks:
        print('\n[error_band] 测试集分段误差区间 (anchor + rock)...')
        fig_error_intervals(args.day,
                            models=('Full_Model', 'BiLSTM', 'SVR'),
                            sensor_kind='anchor')
        fig_error_intervals(args.day,
                            models=('Full_Model', 'BiLSTM', 'SVR'),
                            sensor_kind='rock')

    if 'skill' in tasks:
        print('\n[skill] Skill Score + 动态段指标 (持续基线对比)...')
        days = [int(d) for d in args.skill_days.split(',') if d.strip()]
        gen_skill_tables(days)

    if 'complex' in tasks:
        print('\n[complex] 复杂区 pred vs measured...')
        fig_complex_regions(args.day, BASELINE_GROUP, 'baseline')
        fig_complex_regions(args.day, ABLATION_GROUP, 'ablation')

    print(f'\n[variant={variant}] 完成 ✅')


def _run_multiseed_tables(args):
    """跑完图后 subprocess 调聚合脚本, 产出多 seed 主表 + Table 4 逐天表。

    输出落在 outputs/paper_figures/tables/ (variant 共享), 不重复写。
    """
    import subprocess
    days = args.skill_days  # 复用 --skill_days, 默认 '1,2,3,4,5,6,7'
    py = sys.executable
    cmds = [
        [py, str(ROOT / 'scripts' / 'aggregate_multiseed_r2_pers.py'),
         '--days', days, '--aggregator', 'median'],
        [py, str(ROOT / 'scripts' / 'gen_table_per_day.py'),
         '--days', days, '--mode', 'standard'],
        [py, str(ROOT / 'scripts' / 'gen_table_per_day.py'),
         '--days', days, '--mode', 'persistence'],
    ]
    for cmd in cmds:
        print(f'\n  $ {" ".join(cmd)}')
        env = {**os.environ, 'PYTHONIOENCODING': 'utf-8'}
        try:
            subprocess.run(cmd, check=True, env=env, cwd=str(ROOT))
        except subprocess.CalledProcessError as ex:
            print(f'  [warn] 子脚本失败({cmd[1].split("/")[-1]}): {ex}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--day', type=int, default=1,
                        help='用第几天 (1-7) 跑预测/绘图, 默认1')
    parser.add_argument('--seed', type=int, default=42,
                        help='ckpt 读哪个 seed (multi-seed 实验), 默认 42')
    parser.add_argument('--variant', choices=['main', 'appendix', 'both'],
                        default='both',
                        help='main=6 列基线(论文正文), appendix=9 列基线(附录), both=两套都跑')
    parser.add_argument('--skill_days', type=str, default='1,2,3,4,5,6,7',
                        help='Skill / 动态段 / 多seed 表用哪些天 (逗号分隔), 默认全7天')
    parser.add_argument('--only', nargs='*', default=None,
                        choices=ALL_TASKS,
                        help='只生成指定任务, 默认全部')
    parser.add_argument('--no-predict', action='store_true',
                        help='不重新加载.pth跑预测 (依赖已有缓存)')
    parser.add_argument('--force', action='store_true',
                        help='忽略已有预测缓存, 重新跑')
    parser.add_argument('--skip-multiseed-tables', action='store_true',
                        help='跳过 aggregate_multiseed_r2_pers.py / gen_table_per_day.py 调用')
    args = parser.parse_args()

    setup_matplotlib()
    setup_seed(args.seed)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    variants = (['main', 'appendix'] if args.variant == 'both'
                else [args.variant])
    for v in variants:
        _run_one_variant(args, v)

    # 多 seed 主表 + Table 4 逐天表 (跑一次, 输出到 ROOT_OUT/tables/)
    if not args.skip_multiseed_tables:
        print('\n' + '=' * 70)
        print('生成多 seed 主表 + Table 4 逐天表 (共享, variant 无关)')
        print('=' * 70)
        _run_multiseed_tables(args)

    print('\n' + '=' * 70)
    print('全部完成 ✅')
    print(f'  根: {ROOT_OUT}')
    for v in variants:
        print(f'    {v}/figures/  + {v}/tables/')
    print(f'    tables/        ← 多 seed 主表 + Table 4 逐天表')
    print('=' * 70)


if __name__ == '__main__':
    main()
