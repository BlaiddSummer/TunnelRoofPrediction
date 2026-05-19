import torch
import os

class Config:
    """系统配置类 - 多尺度CNN+双向LSTM+自注意力+GNN（单步预测优化版）"""
    
    # 设备配置
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 模型超参数（5分钟前瞻预测版：温和正则 + 早停更激进）
    HYPERPARAMETERS = {
        'seq_len': 60,  # 输入序列长度（5秒间隔，60步=5分钟）
        'batch_size': 512,
        'epochs': 100,
        'hidden_size': 128,
        'num_layers': 2,
        'dropout': 0.3,         # 0.2 → 0.3 (缓解 train≪val 过拟合)
        'learning_rate': 0.0005, # 1e-3 → 5e-4 (小数据 + 大模型,Adam 偏大易跨过最优)
        'weight_decay': 0.0005, # 1e-4 → 5e-4 (同上)
        # head 末层(预测残差 Δ̂ 的 Linear)单独 wd:把残差推向 0,
        # 等价于 "默认 = persistence,强信号才推开"。
        # 实验证明对参数多的 Full Model 反而是欠拟合的拖累,设 None 不启用。
        # 设为正数 (如 0.01) 可重新开启。
        'head_last_weight_decay': None,
        'patience': 12,         # 25 → 12 (早停更激进, 治后期 plateau)
        'gradient_clip': 1.0,
        # Linear warmup: 前 N 个 epoch 从 0 线性升到 base lr,之后让 plateau 接管。
        # 0 = 不启用 warmup。
        'warmup_epochs': 3,
        # 训练目标：'mse' 与常见论文 Fig.8 一致；'huber' 更抗异常值
        'loss_type': 'mse',
        'huber_delta': 1.0,     # 仅当 loss_type=='huber' 时生效
        # 持续锚定 (residual learning): 模型预测残差 Δ̂ = y[t+H] − y[t],
        # 输出 ŷ = y[t] + Δ̂。 final layer 零初始化 → 训练起点 = persistence,
        # 数学上保证 R²_pers ≥ 0 (在极限),解决"所有模型都输给朴素基线"的问题。
        # 关掉此项可做"是否需要 persistence anchor"消融。
        'use_persistence_anchor': True,
        'lr_schedule': {
            'mode': 'plateau',
            'factor': 0.1,
            'patience': 3,      # 5 → 3 (val plateau 通常 ep15 出现,patience 5 拖到 ep40 才降 lr,过晚)
            'min_lr': 0.0001,
        }
    }
    
    # 多尺度CNN配置
    CNN_CONFIG = {
        'use_cnn': True,  # No-CNN消融：False=关闭CNN，True=启用CNN
        'kernel_sizes': [3, 5, 7],  # 多尺度卷积核
        'num_filters': [32, 64, 128],  # 滤波器数量
    }
    
    # 自注意力配置
    ATTENTION_CONFIG = {
        'num_heads': 8,
        'd_model': 256,
    }
    
    # GNN配置（保留创新点）
    GNN_CONFIG = {
        'use_gnn': True,  # 保留GNN（创新点）
        'hidden_dim': 64,
        'num_layers': 2,
        'use_position': True,  # 是否使用位置信息
        'num_anchor_sensors': 3,  # 左侧锚杆传感器数量（巷道拓扑）
    }
    
    # 预测时间点配置：5 分钟前瞻 (60 步 × 5 秒)
    # 之前是 1 步 (5秒)，太简单 — 持续基线已经几乎完美，所有学习模型反而失分。
    # 改成 60 步后,持续基线 MAE 会上升 ~5x,留给学习模型真正的提升空间。
    PREDICTION_TIMESTEPS = {
        'next': 60,    # 5 分钟 (60 × 5 秒)
        # 'next': 12,  # 1 分钟 (备选)
        # '10min': 120, '30min': 360, '1h': 720,
    }
    
    # 数据配置（使用全部数据；已不包含离层）
    DATA_CONFIG = {
        'data_dir': 'data',
        'file_mapping': {
            '锚杆': ['SQLA7.csv', 'SQLA8.csv', 'SQLA9.csv'],
            # 移除SQLA4、SQLB10、SQLB11、SQLB12（时间有问题）
            '围岩': ['SQLA5.csv', 'SQLA6.csv', 'SQLB9.csv']
        },
        'test_size': 0.2,
        'val_size': 0.1,
        'random_state': 42,
        'time_tolerance': '10s',  # 时间对齐容差
        # 使用全部训练数据（论文方法）
        'sample_ratio': 1.0,  # 从0.3改为1.0，使用全部数据
        'sample_strategy': 'uniform',  # 均匀采样
    }
    
    # 输出路径配置
    OUTPUT_PATHS = {
        'models': 'outputs/models/',
        'results': 'outputs/results/',
        'logs': 'outputs/logs/',
        'plots': 'outputs/plots/'
    }
    
    # 评估指标配置
    EVALUATION_CONFIG = {
        'metrics': ['MAE', 'RMSE', 'MAPE', 'R2', 'MAE_percentage'],
        'save_predictions': True,
        'save_comparison': True,
    }
    
    @staticmethod
    def create_directories():
        """创建必要的输出目录"""
        for path in Config.OUTPUT_PATHS.values():
            os.makedirs(path, exist_ok=True)