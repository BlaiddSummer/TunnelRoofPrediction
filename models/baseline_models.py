# models/baseline_models.py
"""
对比实验基线模型
统一接口：forward(x) -> ({'next': {'锚杆': tensor, '围岩': tensor}}, None)
与 AdvancedPredictionModel 完全兼容，可直接复用 AdvancedTrainer / AdvancedPredictor

所有基线统一接入 persistence-anchor (residual learning),
与完整模型保持公平对比。开关由 config.HYPERPARAMETERS['use_persistence_anchor'] 控制。
"""
import math
import torch
import torch.nn as nn


# ── 公共模块 ───────────────────────────────────────────────────────────────────

class _OutputHead(nn.Module):
    """统一输出头：将 LSTM/Transformer 隐藏态映射为各传感器类型的预测值。
    对称 3 层 MLP,两类传感器结构一致。
    """
    def __init__(self, in_dim: int, hidden_dim: int,
                 num_sensors_per_type: dict, dropout: float = 0.2,
                 zero_init_last: bool = False):
        super().__init__()
        self.heads = nn.ModuleDict()
        for stype, n in num_sensors_per_type.items():
            if n > 0:
                head = nn.Sequential(
                    nn.Linear(in_dim, hidden_dim * 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, n),
                )
                if zero_init_last:
                    nn.init.zeros_(head[-1].weight)
                    nn.init.zeros_(head[-1].bias)
                self.heads[stype] = head

    def forward(self, x: torch.Tensor) -> dict:
        return {stype: head(x) for stype, head in self.heads.items()}


class PositionalEncoding(nn.Module):
    """标准正弦位置编码（Transformer 专用）。"""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 2000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1)])


# ── 公共 mixin: persistence anchor ─────────────────────────────────────────────

class _PersistenceAnchorMixin:
    """所有基线复用的 persistence 残差头。
    forward 子类只需调用 self._anchored_predict(last_hidden, x) 即可。
    """

    def _init_anchor(self, config, num_sensors_per_type):
        self.use_persistence_anchor = config.HYPERPARAMETERS.get(
            'use_persistence_anchor', True
        )
        self._n_anchor = num_sensors_per_type.get('锚杆', 0)
        self._n_rock = num_sensors_per_type.get('围岩', 0)

    def _anchored_predict(self, last_hidden: torch.Tensor, x: torch.Tensor) -> dict:
        """统一的 head + persistence-anchor 出口。self.head 必须返回 dict[sensor_type -> tensor]."""
        out = self.head(last_hidden)
        if not self.use_persistence_anchor:
            return out
        # x 最后一帧 = y[t] (标准化空间),按传感器类型切片做残差锚定
        pers_anchor = x[:, -1, :]                                  # (B, total_sensors)
        pers_per_type = {
            '锚杆': pers_anchor[:, :self._n_anchor],
            '围岩': pers_anchor[:, self._n_anchor:self._n_anchor + self._n_rock],
        }
        anchored = {}
        for stype, delta in out.items():
            if stype in pers_per_type and pers_per_type[stype].shape[1] == delta.shape[1]:
                anchored[stype] = pers_per_type[stype] + delta
            else:
                anchored[stype] = delta
        return anchored


# ── 基线模型 ───────────────────────────────────────────────────────────────────

class LSTMModel(nn.Module, _PersistenceAnchorMixin):
    """基线：标准单向 LSTM"""

    def __init__(self, config, num_sensors_per_type: dict):
        super().__init__()
        total   = sum(num_sensors_per_type.values())
        h       = config.HYPERPARAMETERS['hidden_size']
        drop    = config.HYPERPARAMETERS['dropout']
        nlayers = config.HYPERPARAMETERS['num_layers']

        self._init_anchor(config, num_sensors_per_type)
        self.lstm = nn.LSTM(
            input_size=total,
            hidden_size=h,
            num_layers=nlayers,
            batch_first=True,
            dropout=drop if nlayers > 1 else 0.0
        )
        self.head = _OutputHead(h, h, num_sensors_per_type, drop,
                                zero_init_last=self.use_persistence_anchor)

    def forward(self, x: torch.Tensor):
        out, _ = self.lstm(x)
        last   = out[:, -1, :]
        return {'next': self._anchored_predict(last, x)}, None


class BiLSTMModel(nn.Module, _PersistenceAnchorMixin):
    """基线：双向 LSTM（无 CNN / GNN / Attention）"""

    def __init__(self, config, num_sensors_per_type: dict):
        super().__init__()
        total   = sum(num_sensors_per_type.values())
        h       = config.HYPERPARAMETERS['hidden_size']
        drop    = config.HYPERPARAMETERS['dropout']
        nlayers = config.HYPERPARAMETERS['num_layers']

        self._init_anchor(config, num_sensors_per_type)
        self.bilstm = nn.LSTM(
            input_size=total,
            hidden_size=h,
            num_layers=nlayers,
            batch_first=True,
            dropout=drop if nlayers > 1 else 0.0,
            bidirectional=True
        )
        self.head = _OutputHead(h * 2, h, num_sensors_per_type, drop,
                                zero_init_last=self.use_persistence_anchor)

    def forward(self, x: torch.Tensor):
        out, _ = self.bilstm(x)
        last   = out[:, -1, :]
        return {'next': self._anchored_predict(last, x)}, None


class CNNLSTMModel(nn.Module, _PersistenceAnchorMixin):
    """基线：单尺度 CNN + 单向 LSTM"""

    def __init__(self, config, num_sensors_per_type: dict):
        super().__init__()
        total   = sum(num_sensors_per_type.values())
        h       = config.HYPERPARAMETERS['hidden_size']
        drop    = config.HYPERPARAMETERS['dropout']
        nlayers = config.HYPERPARAMETERS['num_layers']

        self._init_anchor(config, num_sensors_per_type)
        self.cnn = nn.Sequential(
            nn.Conv1d(total, h, kernel_size=3, padding=1),
            nn.BatchNorm1d(h),
            nn.ReLU(),
            nn.Dropout(drop)
        )
        self.lstm = nn.LSTM(
            input_size=h,
            hidden_size=h,
            num_layers=nlayers,
            batch_first=True,
            dropout=drop if nlayers > 1 else 0.0
        )
        self.head = _OutputHead(h, h, num_sensors_per_type, drop,
                                zero_init_last=self.use_persistence_anchor)

    def forward(self, x: torch.Tensor):
        # x: (B, T, F) -> Conv1d 需要 (B, F, T)
        cnn_out  = self.cnn(x.transpose(1, 2)).transpose(1, 2)   # (B, T, h)
        lstm_out, _ = self.lstm(cnn_out)
        last     = lstm_out[:, -1, :]
        return {'next': self._anchored_predict(last, x)}, None


class TransformerModel(nn.Module, _PersistenceAnchorMixin):
    """基线：Transformer Encoder（标准多头自注意力 + 前馈网络）"""

    def __init__(self, config, num_sensors_per_type: dict):
        super().__init__()
        total  = sum(num_sensors_per_type.values())
        h      = config.HYPERPARAMETERS['hidden_size']
        drop   = config.HYPERPARAMETERS['dropout']
        nheads = config.ATTENTION_CONFIG['num_heads']   # 默认 8
        # d_model 必须能被 nheads 整除；h*2=256，8*32 ✓
        d_model = h * 2

        self._init_anchor(config, num_sensors_per_type)
        self.input_proj  = nn.Linear(total, d_model)
        self.pos_enc     = PositionalEncoding(d_model, drop)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nheads,
            dim_feedforward=d_model * 2,
            dropout=drop,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.head = _OutputHead(d_model, h, num_sensors_per_type, drop,
                                zero_init_last=self.use_persistence_anchor)

    def forward(self, x: torch.Tensor):
        x_enc = self.pos_enc(self.input_proj(x))   # (B, T, d_model)
        out = self.transformer(x_enc)
        last = out[:, -1, :]
        return {'next': self._anchored_predict(last, x)}, None
