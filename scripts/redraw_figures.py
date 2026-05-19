# -*- coding: utf-8 -*-
"""
从缓存/表格重绘 最终效果/figures/ 中的 5 张图，改善可读性。
运行: python scripts/redraw_figures.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rcParams

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / 'outputs' / 'paper_figures' / '_cache'
TABLE_DIR  = ROOT / '最终效果' / 'tables'
OUT_DIR    = ROOT / '最终效果' / 'figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 全局风格 ──────────────────────────────────────────────────────────────────
rcParams.update({
    'font.family': ['DejaVu Sans', 'Arial'],
    'axes.unicode_minus': False,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'lines.linewidth': 1.4,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.linestyle': '--',
})

PALETTE = ['#3F6FB6', '#7CB7D6', '#5BAE9D', '#E8A87C', '#C38D9E',
           '#85586F', '#4D8B6F', '#B3823F', '#88498F']
RED = '#C0392B'
BLUE = '#3F6FB6'

DISPLAY = {
    'LSTM': 'LSTM', 'BiLSTM': 'BiLSTM', 'CNN-LSTM': 'CNN-LSTM',
    'Transformer': 'Transformer', 'SVR': 'SVR', 'MLP': 'MLP',
    'Full_Model': 'Ours (Full)',
    'w/o CNN': 'w/o CNN', 'w/o GNN': 'w/o GNN',
    'w/o Attention': 'w/o Attention', 'w/o BiDir': 'w/o BiDir',
}


def save(fig, name: str):
    p = OUT_DIR / f'{name}.png'
    fig.savefig(p, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {p.name}')


def _bar_label(ax, xs, vals, errs=None, fontsize=8, offset_frac=0.06):
    """柱顶/柱底标注数字，自动躲开误差棒顶端。"""
    ylim = ax.get_ylim()
    span = ylim[1] - ylim[0]
    pixel_offset = span * offset_frac
    for i, (xi, v) in enumerate(zip(xs, vals)):
        if np.isnan(v):
            continue
        e = errs[i] if errs is not None and not np.isnan(errs[i]) else 0.0
        if v >= 0:
            y_text = v + e + pixel_offset
            va = 'bottom'
        else:
            y_text = v - e - pixel_offset
            va = 'top'
        txt = f'{v:.2f}±{e:.2f}' if e > 0 else f'{v:.2f}'
        ax.text(xi, y_text, txt, ha='center', va=va, fontsize=fontsize,
                clip_on=False)


def _pad_ylim(ax, top=0.28, bottom=0.10):
    """给柱顶留足数字空间。"""
    lo, hi = ax.get_ylim()
    if ax.get_yscale() == 'log':
        return
    span = hi - lo if hi > lo else max(abs(hi), 1e-3)
    if lo >= 0:
        ax.set_ylim(0, hi + span * top)
    elif hi <= 0:
        ax.set_ylim(lo - span * bottom, 0)
    else:
        ax.set_ylim(lo - span * bottom, hi + span * top)


def _finalize(fig, axes, *, bottom=0.13, top=0.88, hspace=0.50,
              wspace=0.32, legend_ncol=1):
    """把子图各自图例收到整图底部，统一调整间距。"""
    flat = np.ravel(axes)
    # 从第一个有图例的子图收集 handles
    handles, labels = [], []
    for ax in flat:
        leg = ax.get_legend()
        if leg is not None:
            h, l = ax.get_legend_handles_labels()
            if h:
                handles, labels = h, l
            leg.remove()
    if handles:
        fig.legend(handles, labels, loc='lower center',
                   bbox_to_anchor=(0.5, 0.01), ncol=legend_ncol,
                   frameon=True, framealpha=0.9, fontsize=9,
                   edgecolor='#cccccc')
    fig.subplots_adjust(top=top, bottom=bottom, hspace=hspace, wspace=wspace)


# ── 从 npz 缓存计算 MAE ratio（与原始脚本逻辑一致）──────────────────────────

def _mae_ratios_from_cache(model_keys: list, days: list = None):
    """
    model_keys: npz 文件名前缀列表（无 seed 前缀）
    返回 dict: {key: (anc_all, anc_dyn, rock_all, rock_dyn)}，7天平均
    """
    if days is None:
        days = list(range(1, 8))
    from collections import defaultdict
    ar, adr, rr, rdr = (defaultdict(list) for _ in range(4))

    for m in model_keys:
        for d in days:
            pf = CACHE_DIR / f'{m}_day{d}.npz'
            pp = CACHE_DIR / f'_persistence_day{d}.npz'
            if not pf.exists() or not pp.exists():
                continue
            z = np.load(pf)
            p = np.load(pp)
            for key, all_d, dyn_d in [('anchor', ar, adr), ('rock', rr, rdr)]:
                true = z[f'true_{key}']
                pred = z[f'pred_{key}']
                pers = p[f'pred_{key}']
                mae_m = float(np.abs(pred - true).mean())
                mae_p = float(np.abs(pers - true).mean())
                all_d[m].append(mae_m / mae_p if mae_p > 0 else np.nan)
                # dynamic: top-80th-percentile of nonzero change (matches original)
                change = np.abs(true - pers)
                eps = max(1e-8, 1e-4 * float(np.abs(true).mean() or 1.0))
                nonzero = change[change > eps]
                if nonzero.size >= 10:
                    thr = float(np.percentile(nonzero, 80))
                    mask = change >= thr
                    if mask.sum() >= 10:
                        mae_m_d = float(np.abs((pred - true)[mask]).mean())
                        mae_p_d = float(np.abs((pers - true)[mask]).mean())
                        dyn_d[m].append(mae_m_d / mae_p_d if mae_p_d > 0 else np.nan)

    out = {}
    for m in model_keys:
        out[m] = (
            float(np.nanmean(ar[m])),
            float(np.nanmean(adr[m])),
            float(np.nanmean(rr[m])),
            float(np.nanmean(rdr[m])),
        )
    return out


def _draw_mae_panels(axes, group_keys, display_names, ratios_dict, fig_title,
                     use_ablation_colors=False):
    """4-panel MAE ratio bar chart, shared logic for fig1 and fig5."""
    panels = [
        ('Anchor — All samples',     0, False),
        ('Anchor — Dynamic top 20%', 1, False),
        ('Rock — All samples',       2, True),
        ('Rock — Dynamic top 20%',   3, False),
    ]
    for ax, (title, idx, use_log) in zip(axes.flat, panels):
        vals = np.array([ratios_dict[m][idx] for m in group_keys], dtype=float)
        x = np.arange(len(group_keys))
        if use_ablation_colors:
            colors = [PALETTE[i % len(PALETTE)] for i in range(len(group_keys))]
        else:
            colors = [RED if (not np.isnan(v) and v > 1) else PALETTE[i % len(PALETTE)]
                      for i, v in enumerate(vals)]
        ax.bar(x, vals, color=colors, width=0.62, zorder=2)
        ax.axhline(1.0, color='black', linewidth=1.2, linestyle='--',
                   label='Persistence baseline', zorder=1)
        ax.set_xticks(x)
        ax.set_xticklabels(display_names, rotation=30, ha='right', fontsize=9)
        ax.set_ylabel('MAE / MAE_persistence', fontsize=10)
        ax.set_title(title, fontsize=10, pad=4)
        if use_log:
            ax.set_yscale('log')
        ax.grid(axis='y', linestyle=':', alpha=0.35)
        if not use_log:
            _pad_ylim(ax, top=0.30, bottom=0.08)
        for xi, v in zip(x, vals):
            if np.isnan(v):
                continue
            if use_log:
                ax.text(xi, v * 1.15, f'{v:.2f}', ha='center', va='bottom',
                        fontsize=8, clip_on=False)
            else:
                ylim = ax.get_ylim()
                off = (ylim[1] - ylim[0]) * 0.04
                y_text = v + off if v >= 0 else v - off
                ax.text(xi, y_text, f'{v:.2f}', ha='center',
                        va='bottom' if v >= 0 else 'top', fontsize=8, clip_on=False)


# ══════════════════════════════════════════════════════════════════════════════
# 图 1: MAE relative to persistence — Baseline (baseline group)
# ══════════════════════════════════════════════════════════════════════════════

def fig1_mae_baseline():
    keys = ['LSTM', 'BiLSTM', 'CNN-LSTM', 'Transformer', 'Full_Model']
    names = ['LSTM', 'BiLSTM', 'CNN-LSTM', 'Transformer', 'Ours (Full)']
    ratios = _mae_ratios_from_cache(keys)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    fig.suptitle('Fig. MAE relative to persistence baseline (baseline group; < 1 = better than persistence)',
                 fontsize=11, y=0.99)
    _draw_mae_panels(axes, keys, names, ratios,
                     fig_title='', use_ablation_colors=False)
    _finalize(fig, axes, bottom=0.14, top=0.93, hspace=0.60, wspace=0.35, legend_ncol=1)
    save(fig, '1')


# ══════════════════════════════════════════════════════════════════════════════
# 图 2: 时序对比 + 散点 (Anchor #2, day1)
# ══════════════════════════════════════════════════════════════════════════════

def fig2_compare_timeline():
    models = ['Full_Model', 'BiLSTM', 'CNN-LSTM', 'Transformer']
    label_map = {
        'Full_Model': 'Ours (Full)', 'BiLSTM': 'BiLSTM',
        'CNN-LSTM': 'CNN-LSTM', 'Transformer': 'Transformer',
    }

    preds, trues = {}, {}
    for m in models:
        z = np.load(CACHE_DIR / f'{m}_day1.npz')
        preds[m] = z['pred_anchor'][:, 1]   # Anchor #2 (idx=1)
        trues[m] = z['true_anchor'][:, 1]
    true = trues[models[0]]

    def _smooth(a, k=60):
        return pd.Series(a).rolling(window=k, center=True, min_periods=1).mean().values

    lo = float(min(true.min(), min(v.min() for v in preds.values())))
    hi = float(max(true.max(), max(v.max() for v in preds.values())))

    n_models = len(models)
    fig = plt.figure(figsize=(13.5, 8.2))
    gs = gridspec.GridSpec(
        2, n_models,
        height_ratios=[1.15, 0.90],
        hspace=0.52, wspace=0.30,
        top=0.92, bottom=0.13, left=0.07, right=0.98,
    )

    # 顶部时序图 (占满一行)
    ax_top = fig.add_subplot(gs[0, :])
    ax_top.plot(np.arange(len(true)), _smooth(true), color='black',
                linewidth=1.7, label='Measured', zorder=10)
    for i, m in enumerate(models):
        ax_top.plot(np.arange(len(preds[m])), _smooth(preds[m]),
                    color=PALETTE[i % len(PALETTE)],
                    linewidth=1.1, alpha=0.85, label=label_map[m])
    ax_top.set_xlabel('Test sample index')
    ax_top.set_ylabel('Value')
    ax_top.set_title('(a) Time series comparison — Anchor #2', fontsize=10)
    ax_top.margins(x=0.01)
    # 图例放在子图右上角之外，避免遮盖曲线
    ax_top.legend(loc='upper left', ncol=n_models + 1,
                  frameon=True, framealpha=0.85, fontsize=8.5,
                  edgecolor='#cccccc', bbox_to_anchor=(0.0, 1.0))

    # 底部散点图
    scatter_letters = 'bcde'
    for i, m in enumerate(models):
        ax = fig.add_subplot(gs[1, i])
        y = preds[m]
        r2 = 1.0 - np.sum((y - true) ** 2) / np.sum((true - true.mean()) ** 2)
        ax.scatter(true, y, s=6, alpha=0.40, color=PALETTE[i % len(PALETTE)],
                   edgecolor='none', rasterized=True)
        ax.plot([lo, hi], [lo, hi], color='black', linewidth=1.0, linestyle='--')
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(f'({scatter_letters[i]}) {label_map[m]}  R² = {r2:.4f}',
                     fontsize=9, pad=3)
        ax.set_xlabel('Measured', fontsize=9)
        if i == 0:
            ax.set_ylabel('Predicted', fontsize=9)

    fig.text(0.5, 0.99, 'Fig. Comparison of models on Dataset-1',
             ha='center', va='top', fontsize=11, transform=fig.transFigure)
    save(fig, '2')


# ══════════════════════════════════════════════════════════════════════════════
# 图 3: Persistence-relative R² — Ablation (median ± MAD)
# ══════════════════════════════════════════════════════════════════════════════

def fig3_r2_ablation():
    df = pd.read_csv(TABLE_DIR / 'table_r2_pers_multiseed_ablation_median.csv',
                     index_col=0)
    group_raw = ['Ours (Full)', 'w/o CNN', 'w/o GNN', 'w/o Attention', 'w/o BiDir']
    group = [g for g in group_raw if g in df.index]

    panels = [
        ('(a) Anchor — All samples',    'Anchor_R2pers_mean',     'Anchor_R2pers_std'),
        ('(b) Anchor — Dynamic top 20%','Anchor_R2pers_dyn_mean', 'Anchor_R2pers_dyn_std'),
        ('(c) Rock — All samples',      'Rock_R2pers_mean',       'Rock_R2pers_std'),
        ('(d) Rock — Dynamic top 20%',  'Rock_R2pers_dyn_mean',   'Rock_R2pers_dyn_std'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 7.8))
    fig.suptitle(r'Fig. Persistence-relative $R^2$ (median ± MAD across seeds) — 5 seeds × ablation',
                 fontsize=11, y=0.99)

    for ax, (title, col_m, col_s) in zip(axes.flat, panels):
        means = df.loc[group, col_m].values.astype(float)
        stds  = df.loc[group, col_s].values.astype(float)
        x = np.arange(len(group))
        colors = [RED if g == 'Ours (Full)' else BLUE for g in group]
        ax.bar(x, means, yerr=stds, color=colors, width=0.60,
               capsize=5, ecolor='#444444',
               error_kw={'elinewidth': 1.2, 'capthick': 1.2}, zorder=2)
        ax.axhline(0.0, color='black', linewidth=1.1, linestyle='--',
                   label=r'Persistence ($R^2$ = 0)', zorder=1)
        ax.set_xticks(x)
        ax.set_xticklabels(group, rotation=28, ha='right', fontsize=9)
        ax.set_ylabel(r'$R^2_{\mathrm{pers}}$ (median ± MAD)', fontsize=9)
        ax.set_title(title, fontsize=10, pad=4)
        ax.grid(axis='y', linestyle=':', alpha=0.40)
        _pad_ylim(ax, top=0.30, bottom=0.12)
        _bar_label(ax, x, means, stds, fontsize=7.5, offset_frac=0.04)

    _finalize(fig, axes, bottom=0.13, top=0.93, hspace=0.62, wspace=0.36,
              legend_ncol=1)
    save(fig, '3')


# ══════════════════════════════════════════════════════════════════════════════
# 图 4: Persistence-relative R² — Baseline (median ± MAD)
# ══════════════════════════════════════════════════════════════════════════════

def fig4_r2_baseline():
    df = pd.read_csv(TABLE_DIR / 'table_r2_pers_multiseed_baseline_median.csv',
                     index_col=0)
    group_raw = ['LSTM', 'BiLSTM', 'CNN-LSTM', 'SVR', 'MLP', 'Ours (Full)']
    group = [g for g in group_raw if g in df.index]

    panels = [
        ('(a) Anchor — All samples',    'Anchor_R2pers_mean',     'Anchor_R2pers_std'),
        ('(b) Anchor — Dynamic top 20%','Anchor_R2pers_dyn_mean', 'Anchor_R2pers_dyn_std'),
        ('(c) Rock — All samples',      'Rock_R2pers_mean',       'Rock_R2pers_std'),
        ('(d) Rock — Dynamic top 20%',  'Rock_R2pers_dyn_mean',   'Rock_R2pers_dyn_std'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 7.8))
    fig.suptitle(r'Fig. Persistence-relative $R^2$ (median ± MAD across seeds) — 5 seeds × baseline',
                 fontsize=11, y=0.99)

    for ax, (title, col_m, col_s) in zip(axes.flat, panels):
        means = df.loc[group, col_m].values.astype(float)
        stds  = df.loc[group, col_s].values.astype(float)
        x = np.arange(len(group))
        colors = [RED if g == 'Ours (Full)' else BLUE for g in group]
        ax.bar(x, means, yerr=stds, color=colors, width=0.60,
               capsize=5, ecolor='#444444',
               error_kw={'elinewidth': 1.2, 'capthick': 1.2}, zorder=2)
        ax.axhline(0.0, color='black', linewidth=1.1, linestyle='--',
                   label=r'Persistence ($R^2$ = 0)', zorder=1)
        ax.set_xticks(x)
        ax.set_xticklabels(group, rotation=28, ha='right', fontsize=9)
        ax.set_ylabel(r'$R^2_{\mathrm{pers}}$ (median ± MAD)', fontsize=9)
        ax.set_title(title, fontsize=10, pad=4)
        ax.grid(axis='y', linestyle=':', alpha=0.40)
        _pad_ylim(ax, top=0.30, bottom=0.12)
        _bar_label(ax, x, means, stds, fontsize=7.5, offset_frac=0.04)

    _finalize(fig, axes, bottom=0.13, top=0.93, hspace=0.62, wspace=0.36,
              legend_ncol=1)
    save(fig, '4')


# ══════════════════════════════════════════════════════════════════════════════
# 图 5: MAE relative to persistence — Ablation
# ══════════════════════════════════════════════════════════════════════════════

def fig5_mae_ablation():
    keys = ['Full_Model', 'w_o_CNN', 'w_o_GNN', 'w_o_Attention', 'w_o_BiDir']
    names = ['Ours (Full)', 'w/o CNN', 'w/o GNN', 'w/o Attention', 'w/o BiDir']
    ratios = _mae_ratios_from_cache(keys)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    fig.suptitle('Fig. MAE relative to persistence baseline (ablation group; < 1 = better than persistence)',
                 fontsize=11, y=0.99)
    _draw_mae_panels(axes, keys, names, ratios,
                     fig_title='', use_ablation_colors=True)
    _finalize(fig, axes, bottom=0.14, top=0.93, hspace=0.60, wspace=0.35, legend_ncol=1)
    save(fig, '5')


# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print(f'Output dir: {OUT_DIR}')
    print('Drawing fig 1 ...')
    fig1_mae_baseline()
    print('Drawing fig 2 ...')
    fig2_compare_timeline()
    print('Drawing fig 3 ...')
    fig3_r2_ablation()
    print('Drawing fig 4 ...')
    fig4_r2_baseline()
    print('Drawing fig 5 ...')
    fig5_mae_ablation()
    print('Done.')
