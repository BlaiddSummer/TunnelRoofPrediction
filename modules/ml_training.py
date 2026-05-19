# modules/ml_training.py
"""
经典 ML 基线 (RF/XGBoost/SVR/MLP) 的训练包装。
接口与 modules.training.AdvancedTrainer 一致, run_one 里可直接替换:

    trainer = MLTrainer(config, model, train_loader, val_loader)
    model, history = trainer.train()
"""
from __future__ import annotations

import os
import time
import numpy as np
import torch.utils.data as Data

from models.ml_baselines import save_ml_baseline


def _loader_to_numpy(loader: Data.DataLoader) -> tuple[np.ndarray, dict]:
    """把 MultiTimeStepDataset 的 DataLoader 拼回 numpy 形式。"""
    xs = []
    ys: dict = {}
    for batch_X, batch_y in loader:
        xs.append(batch_X.numpy())
        if 'next' in batch_y:
            for st, t in batch_y['next'].items():
                if t is None:
                    continue
                ys.setdefault(st, []).append(t.numpy())
    X = np.concatenate(xs, axis=0) if xs else np.zeros((0,))
    y_dict = {'next': {st: np.concatenate(lst, axis=0)
                       for st, lst in ys.items() if lst}}
    return X, y_dict


class MLTrainer:
    """ML 基线训练器: fit 一次, 没有 epoch / lr schedule / early stopping (子模型内部已处理)。"""

    def __init__(self, config, model, train_loader, val_loader):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

    def train(self):
        print('=' * 80)
        print(f'🚀 开始训练 ML 基线: {getattr(self.model, "DISPLAY_NAME", type(self.model).__name__)}')
        print('=' * 80)
        t0 = time.time()

        train_X, train_y = _loader_to_numpy(self.train_loader)
        val_X,   val_y   = _loader_to_numpy(self.val_loader)
        print(f'  训练样本: {train_X.shape}, 验证样本: {val_X.shape}')

        self.model.fit(train_X, train_y, val_X=val_X, val_y=val_y)

        # 保存 .pkl checkpoint (与 NN 的 best_model.pth 平行)
        ckpt = os.path.join(self.config.OUTPUT_PATHS['models'], 'best_model.pkl')
        save_ml_baseline(self.model, ckpt)
        print(f'  ✅ 模型已保存: {ckpt}')

        print(f'✅ 训练完成, 耗时 {(time.time() - t0) / 60:.2f} 分钟')
        # history 为 None: 下游写文件容忍这个空值
        return self.model, None
