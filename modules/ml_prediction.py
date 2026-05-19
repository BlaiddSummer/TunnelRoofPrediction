# modules/ml_prediction.py
"""ML 基线推理: 接口与 AdvancedPredictor.predict_batch 一致。"""
from __future__ import annotations

import numpy as np
import torch.utils.data as Data

from modules.ml_training import _loader_to_numpy


class MLPredictor:
    """跑 ML baseline 的预测, 返回与 AdvancedPredictor 完全相同的字典结构。"""

    def __init__(self, config, model, scalers):
        self.config = config
        self.model = model
        self.scalers = scalers

    def predict_batch(self, data_loader: Data.DataLoader):
        X, y_dict = _loader_to_numpy(data_loader)
        if X.size == 0:
            return {'next': {}}, {'next': {}}
        preds = self.model.predict(X)
        # y_dict 已经是 numpy
        actuals = {'next': {st: np.asarray(arr)
                            for st, arr in y_dict.get('next', {}).items()}}
        return preds, actuals
