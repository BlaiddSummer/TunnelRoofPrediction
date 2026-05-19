# -*- coding: utf-8 -*-
"""论文图统一排版：单图例、y 轴留白（贴 0）、柱图 x 范围。"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def _is_nan(v) -> bool:
    try:
        return v is None or (isinstance(v, float) and np.isnan(v))
    except TypeError:
        return False


def pad_ylim(ax, *, top: float = 0.20, bottom: float = 0.08,
             stick_zero: bool = True) -> None:
    """加大 y 留白；全正/全负子图时让 0 线贴坐标轴（柱底/柱顶不悬空）。"""
    ymin, ymax = ax.get_ylim()
    if ymax <= ymin:
        span = max(abs(float(ymax)), abs(float(ymin)), 1.0)
        ymin, ymax = ymin - span * 0.05, ymax + span * 0.05

    if ax.get_yscale() == 'log':
        if ymin <= 0:
            pos = [t for t in ax.get_yticks() if t > 0]
            ymin = pos[0] if pos else 1e-3
        log_span = np.log10(ymax) - np.log10(ymin)
        if log_span <= 0:
            log_span = 0.5
        ax.set_ylim(ymin / 10 ** (bottom * log_span),
                    ymax * 10 ** (top * log_span))
        return

    span = ymax - ymin if ymax > ymin else max(abs(ymax), 1.0)

    if ymin < 0 < ymax:
        new_ymin = ymin - span * bottom
        new_ymax = ymax + span * top
    elif ymin >= 0:
        new_ymin = 0.0 if stick_zero else ymin
        new_ymax = ymax + span * top
    else:
        new_ymin = ymin - span * bottom
        new_ymax = 0.0 if stick_zero else ymax

    ax.set_ylim(new_ymin, new_ymax)


def annotate_bar_values(ax, xs, ys, *, errs=None, fmt: str = '{:.2f}',
                      fontsize: float = 8) -> None:
    ys = list(ys)
    errs = list(errs) if errs is not None else None
    for i, (xi, y) in enumerate(zip(xs, ys)):
        if _is_nan(y):
            continue
        y = float(y)
        err = 0.0
        if errs is not None and i < len(errs) and not _is_nan(errs[i]):
            err = float(errs[i])
        if err:
            y_text = y + err if y >= 0 else y - err
            label = f'{y:.2f}±{err:.2f}'
        else:
            y_text = y
            label = fmt.format(y) if '{' in fmt else fmt
        va = 'bottom' if y >= 0 else 'top'
        ax.text(xi, y_text, label, ha='center', va=va, fontsize=fontsize)


def figure_legend_below_title(fig, axes, *, ncol: int = 1,
                              fontsize: float = 9, y: float = 0.97) -> None:
    flat = np.ravel(axes)
    for ax in flat:
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()

    handles, labels = flat[0].get_legend_handles_labels()
    if not handles:
        return

    fig.legend(handles, labels, loc='upper center',
               bbox_to_anchor=(0.5, y), ncol=ncol,
               frameon=False, fontsize=fontsize)


def finalize_bar_grid(fig, axes, *, top: float = 0.88, bottom: float = 0.11,
                      hspace: float = 0.42, wspace: float = 0.30,
                      legend_y: float = 0.99, pad_top: float = 0.20,
                      pad_bottom: float = 0.08) -> None:
    """y 留白 + 贴 0 + 收紧 x + 整图单图例。"""
    for ax in np.ravel(axes):
        n = len(ax.get_xticks())
        if n > 0:
            ax.set_xlim(-0.6, n - 0.4)
        pad_ylim(ax, top=pad_top, bottom=pad_bottom, stick_zero=True)

    figure_legend_below_title(fig, axes, y=legend_y)
    fig.subplots_adjust(top=top, bottom=bottom, hspace=hspace, wspace=wspace)
