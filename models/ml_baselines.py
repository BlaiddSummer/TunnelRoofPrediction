# models/ml_baselines.py
"""
经典机器学习对比基线: RandomForest / XGBoost / SVR / MLP (sklearn).

统一接口:
  fit(train_X, train_y, val_X=None, val_y=None)
  predict(X) -> {'next': {'锚杆': np.ndarray (N, n_anchor), '围岩': ...}}
  state_dict() / load_state_dict(state)
  parameters() -> []   (兼容 sum(p.numel()) 调用,ML 模型当 0 参数报)

输入张量:
  X shape (N, seq_len, total_features), 标准化空间
  y_dict['next']['锚杆'] shape (N, n_anchor), 标准化空间

模型设计:
  - 输入 flat: (N, seq_len * total_features). 用 seq_len=60 / 6 sensor → 360 维。
  - 与 NN baseline 一致接入 persistence anchor (residual learning):
    每个 sensor 各一个回归器, 预测 Δ̂ = y[t+H] - y[t], 输出 ŷ = y[t] + Δ̂。
    "y[t]" = X[:, -1, sensor_col_idx], 即输入最后一帧。
  - 这样所有 baseline (NN + ML) 都站在同样的 persistence anchor 之上, 公平对比。

依赖:
  scikit-learn  (必装, RF/SVR/MLP 都来自这里)
  xgboost       (XGBoost baseline 用; 未装则该 baseline 不可用)
"""
from __future__ import annotations

import os
import pickle
from typing import Any

import numpy as np

# 这些 import 是必须的; 服务器上请保证已装。
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor

try:
    import xgboost as xgb
    _HAS_XGB = True
except ImportError:        # 未装 xgboost 时退化, 让其他三个 baseline 仍可用
    _HAS_XGB = False


def _flatten(X: np.ndarray) -> np.ndarray:
    """(N, seq_len, F) -> (N, seq_len*F). 兼容已经是 2D 的情况。"""
    X = np.asarray(X)
    if X.ndim == 3:
        return X.reshape(X.shape[0], -1)
    return X


def _split_targets(y_dict: dict, sensor_types: list[str]) -> dict:
    """从训练管线的 y_dict 拿出 {sensor_type: (N, n_sensors)} ndarray。"""
    out = {}
    nxt = y_dict.get('next', {}) if isinstance(y_dict, dict) else {}
    for st in sensor_types:
        arr = nxt.get(st, None)
        if arr is None:
            continue
        arr = np.asarray(arr)
        if arr.size == 0:
            continue
        out[st] = arr
    return out


def _persistence_targets(X: np.ndarray, sensor_type_indices: dict,
                         sensor_types: list[str]) -> dict:
    """对每个 sensor type 切出 y[t] = X[:, -1, sensor_cols], 用于 anchor。"""
    last_frame = X[:, -1, :]                                      # (N, F)
    out = {}
    for st in sensor_types:
        if st not in sensor_type_indices:
            continue
        s, e = sensor_type_indices[st]
        out[st] = last_frame[:, s:e]
    return out


class _MLBaselineBase:
    """4 个 ML baseline 的通用骨架。
    子类只需在 _make_regressor() 返回一个 (n_outputs=1) 的回归器,
    我们自动 wrap 成 MultiOutputRegressor 处理多 sensor。

    对每个 sensor type 各训一个 MultiOutputRegressor, 这样 anchor 切片对齐很自然。
    """

    DISPLAY_NAME: str = 'ML-Baseline'
    SENSOR_TYPES = ('锚杆', '围岩')

    def __init__(self, config, num_sensors_per_type: dict,
                 sensor_type_indices: dict | None = None,
                 seed: int = 42):
        self.config = config
        self.num_sensors_per_type = num_sensors_per_type
        self.sensor_type_indices = sensor_type_indices
        self.seed = seed
        self.use_persistence_anchor = bool(
            config.HYPERPARAMETERS.get('use_persistence_anchor', True)
        )
        # 每种 sensor 一个回归器
        self.regressors: dict[str, Any] = {}
        # 训练完后再填: y[t] 切片下标 (start, end), 用于 predict 时切 persistence anchor
        self._sti = sensor_type_indices

    # ── 子类实现 ───────────────────────────────────────────────────────────
    def _make_regressor(self):
        raise NotImplementedError

    # ── 公共 fit / predict ────────────────────────────────────────────────
    def fit(self, train_X: np.ndarray, train_y: dict,
            val_X: np.ndarray | None = None, val_y: dict | None = None):
        del val_X, val_y                            # 多数 ML baseline 无 early stop, 忽略
        flatX = _flatten(train_X)
        ys = _split_targets(train_y, list(self.SENSOR_TYPES))

        if self.use_persistence_anchor and self._sti is not None:
            pers = _persistence_targets(train_X, self._sti, list(self.SENSOR_TYPES))
        else:
            pers = {}

        for st, y in ys.items():
            target = y - pers[st] if (st in pers) else y           # 学残差 Δ̂
            base = self._make_regressor()
            # 多输出: SVR / 部分模型不原生支持, 统一 MultiOutputRegressor
            reg = MultiOutputRegressor(base)
            reg.fit(flatX, target)
            self.regressors[st] = reg
            print(f'  [{self.DISPLAY_NAME}] {st} 训练完成: '
                  f'X={flatX.shape}, y={target.shape}')
        return self

    def predict(self, X: np.ndarray) -> dict:
        flatX = _flatten(X)
        out = {'next': {}}
        pers = (_persistence_targets(X, self._sti, list(self.SENSOR_TYPES))
                if (self.use_persistence_anchor and self._sti is not None) else {})
        for st, reg in self.regressors.items():
            delta = reg.predict(flatX)
            if st in pers:
                pred = pers[st] + delta
            else:
                pred = delta
            out['next'][st] = pred.astype(np.float32)
        return out

    # ── 兼容 nn.Module 的接口 (供 experiment_main / aggregate 复用) ────────
    def state_dict(self) -> dict:
        return {
            'regressors': self.regressors,
            'sensor_type_indices': self._sti,
            'use_persistence_anchor': self.use_persistence_anchor,
        }

    def load_state_dict(self, state: dict):
        self.regressors = state.get('regressors', {})
        self._sti = state.get('sensor_type_indices', self._sti)
        self.use_persistence_anchor = state.get(
            'use_persistence_anchor', self.use_persistence_anchor
        )
        return self

    def parameters(self):                                          # noqa: D401
        return iter([])                                            # 让 sum(p.numel()) = 0

    def to(self, device):                                          # nn.Module 兼容
        return self

    def eval(self):                                                # nn.Module 兼容
        return self

    def train_mode(self):                                          # 避免与 trainer.train() 同名
        return self


# ── 4 个具体 baseline ────────────────────────────────────────────────────


class RandomForestBaseline(_MLBaselineBase):
    DISPLAY_NAME = 'RF'

    def _make_regressor(self):
        return RandomForestRegressor(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=self.seed,
        )


class XGBoostBaseline(_MLBaselineBase):
    DISPLAY_NAME = 'XGB'

    def _make_regressor(self):
        if not _HAS_XGB:
            raise ImportError(
                'xgboost 未安装。请 `pip install xgboost` 或从 EXPERIMENTS 里移除 XGBoost。'
            )
        return xgb.XGBRegressor(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            tree_method='hist',
            n_jobs=-1,
            random_state=self.seed,
            verbosity=0,
        )


class SVRBaseline(_MLBaselineBase):
    DISPLAY_NAME = 'SVR'

    def _make_regressor(self):
        # RBF kernel, 标准化空间下 C=1.0 是合理起点。SVR 慢, fit ~ O(N²)。
        return SVR(kernel='rbf', C=1.0, epsilon=0.01, gamma='scale')


class MLPBaseline(_MLBaselineBase):
    DISPLAY_NAME = 'MLP'

    def _make_regressor(self):
        return MLPRegressor(
            hidden_layer_sizes=(256, 128, 64),
            activation='relu',
            solver='adam',
            alpha=1e-4,
            batch_size=256,
            learning_rate_init=1e-3,
            max_iter=200,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
            random_state=self.seed,
            verbose=False,
        )


# ── 保存 / 加载: 不走 torch.save, 直接 pickle ───────────────────────────────

def save_ml_baseline(model: _MLBaselineBase, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(model.state_dict(), f, protocol=pickle.HIGHEST_PROTOCOL)


def load_ml_baseline_state(path: str) -> dict:
    with open(path, 'rb') as f:
        return pickle.load(f)


ML_BASELINE_REGISTRY = {
    'RF':       RandomForestBaseline,
    'XGBoost':  XGBoostBaseline,
    'SVR':      SVRBaseline,
    'MLP':      MLPBaseline,
}
