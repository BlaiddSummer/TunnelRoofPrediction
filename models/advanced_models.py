import torch
import torch.nn as nn
import torch.nn.functional as F
import math

try:
    from torch_geometric.nn import GCNConv
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    print("[WARNING] torch_geometric not installed, GNN will use SimpleGNN fallback")


class MultiScaleCNN(nn.Module):
    """多尺度CNN特征提取"""
    def __init__(self, input_size, kernel_sizes=[3, 5, 7], num_filters=[64, 128, 256], dropout=0.3):
        super(MultiScaleCNN, self).__init__()
        self.kernel_sizes = kernel_sizes
        self.num_filters = num_filters
        
        # 为每个尺度创建卷积层
        self.conv_layers = nn.ModuleList()
        for kernel_size, num_filter in zip(kernel_sizes, num_filters):
            conv = nn.Sequential(
                nn.Conv1d(input_size, num_filter, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(num_filter),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv1d(num_filter, num_filter, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(num_filter),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            self.conv_layers.append(conv)
        
        # 融合层
        total_filters = sum(num_filters)
        self.fusion = nn.Sequential(
            nn.Conv1d(total_filters, total_filters, 1),
            nn.BatchNorm1d(total_filters),
            nn.ReLU()
        )
        
    def forward(self, x):
        # x: (batch, seq_len, features) -> (batch, features, seq_len)
        x = x.transpose(1, 2)
        
        # 多尺度特征提取
        multi_scale_features = []
        for conv in self.conv_layers:
            feature = conv(x)
            multi_scale_features.append(feature)
        
        # 拼接多尺度特征
        concat_features = torch.cat(multi_scale_features, dim=1)
        
        # 融合
        fused = self.fusion(concat_features)
        
        # 转回 (batch, seq_len, features)
        return fused.transpose(1, 2)


class SelfAttention(nn.Module):
    """自注意力机制"""
    def __init__(self, d_model, num_heads=8, dropout=0.1):
        super(SelfAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"
        
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.size()
        residual = x
        
        # Q, K, V
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        output = self.output(attn_output)
        output = self.layer_norm(output + residual)
        
        return output, attn_weights


class SimpleGNN(nn.Module):
    """简化的GNN实现（当torch_geometric不可用时）— per-sensor MLP，共享权重。
    输出与 SensorGNN(torch_geometric) 一致: (B, T, N*hidden_dim)。
    """
    def __init__(self, num_sensors, hidden_dim=64, num_layers=2):
        super(SimpleGNN, self).__init__()
        self.num_sensors = num_sensors
        self.hidden_dim = hidden_dim

        # 每个传感器独立通过相同 MLP；输入是单个标量。
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(1, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))

    def forward(self, x):
        # x: (batch, seq_len, num_sensors)
        batch_size, seq_len, num_sensors = x.size()
        h = x.reshape(batch_size * seq_len * num_sensors, 1)
        for layer in self.layers:
            h = F.relu(layer(h))
        return h.reshape(batch_size, seq_len, num_sensors * self.hidden_dim)


class SensorGNN(nn.Module):
    """传感器图神经网络（巷道拓扑：左侧锚杆 + 顶部围岩，三组配对）

    巷道示意图（沿巷道方向）：
        顶部: [围岩0]---[围岩1]---[围岩2]
                |          |         |
        左侧: [锚杆0]---[锚杆1]---[锚杆2]

    节点排列约定（与 data_utils 加载顺序一致）：
        0 ~ num_anchor-1       : 锚杆传感器
        num_anchor ~ total-1   : 围岩传感器
    """
    def __init__(self, num_sensors, hidden_dim=64, num_layers=2, use_position=True, num_anchor=None):
        super(SensorGNN, self).__init__()
        self.num_sensors = num_sensors
        self.use_position = use_position
        self.num_anchor = num_anchor if num_anchor is not None else num_sensors // 2

        if TORCH_GEOMETRIC_AVAILABLE:
            # 使用torch_geometric
            self.gnn_layers = nn.ModuleList()
            self.gnn_layers.append(GCNConv(1, hidden_dim))
            for _ in range(num_layers - 1):
                self.gnn_layers.append(GCNConv(hidden_dim, hidden_dim))

            # 巷道位置
            if use_position:
                positions = self._create_tunnel_positions(num_sensors)
                self.register_buffer('positions', positions)

            # 巷道拓扑边
            edge_index = self._build_tunnel_edges(num_sensors)
            self.register_buffer('edge_index', edge_index)
            self.use_geometric = True
        else:
            # 使用简化版本
            self.gnn = SimpleGNN(num_sensors, hidden_dim, num_layers)
            self.use_geometric = False

    def _create_tunnel_positions(self, num_sensors):
        """巷道坐标：(沿巷道位置, 壁面侧)
           锚杆: x=0（左侧），围岩: x=1（顶部）
        """
        n_anchor = self.num_anchor
        n_rock = num_sensors - n_anchor
        positions = []
        for i in range(n_anchor):
            positions.append([float(i), 0.0])
        for i in range(n_rock):
            positions.append([float(i), 1.0])
        return torch.tensor(positions, dtype=torch.float)

    def _build_tunnel_edges(self, num_sensors):
        """巷道拓扑边（双向）：
           1. 同组锚杆-围岩配对（垂直连接）
           2. 沿巷道方向锚杆相邻
           3. 沿巷道方向围岩相邻
        """
        n_anchor = self.num_anchor
        n_rock = num_sensors - n_anchor
        n_pairs = min(n_anchor, n_rock)
        edges = []

        # 1. 同组配对：锚杆i ↔ 围岩i
        for i in range(n_pairs):
            edges += [[i, n_anchor + i], [n_anchor + i, i]]

        # 2. 沿巷道锚杆相邻：锚杆i ↔ 锚杆i+1
        for i in range(n_anchor - 1):
            edges += [[i, i + 1], [i + 1, i]]

        # 3. 沿巷道围岩相邻：围岩i ↔ 围岩i+1
        for i in range(n_rock - 1):
            edges += [[n_anchor + i, n_anchor + i + 1],
                      [n_anchor + i + 1, n_anchor + i]]

        return torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    def forward(self, x):
        # x: (batch, seq_len, num_sensors)
        batch_size, seq_len, num_sensors = x.size()

        if self.use_geometric:
            # Flatten batch and time into one graph-batch dimension so all
            # (batch * seq_len) graphs are processed in a single GCN call.
            num_graphs = batch_size * seq_len

            # Node features: (G*N, 1)  where G = num_graphs, N = num_sensors
            x_nodes = x.reshape(num_graphs * num_sensors, 1)

            # Build batched edge_index: replicate edges for each graph with a
            # per-graph node offset so indices don't collide.
            # self.edge_index: (2, E) -> (2, G*E)
            offsets = torch.arange(num_graphs, device=x.device) * num_sensors  # (G,)
            edge_index_batched = (
                self.edge_index.unsqueeze(1)        # (2, 1, E)
                + offsets.view(1, num_graphs, 1)    # broadcast -> (2, G, E)
            ).reshape(2, -1)                        # (2, G*E)

            h = x_nodes
            for gcn in self.gnn_layers:
                h = gcn(h, edge_index_batched)
                h = F.relu(h)

            # Per-sensor readout: (G*N, hidden) -> (G, N, hidden) -> (G, N*hidden)
            # 不再 mean-pool 把节点糊在一起；保留每个传感器自己的表示给下游。
            # Why: mean-pool 会丢掉锚杆/围岩的差异化信息，对围岩 MAE 不利。
            hidden_dim = h.shape[-1]
            return h.reshape(batch_size, seq_len, num_sensors * hidden_dim)
        else:
            return self.gnn(x)


class AdvancedPredictionModel(nn.Module):
    """多尺度CNN(可关闭) + GNN + 双向LSTM + 自注意力 预测模型（单步预测版）"""
    def __init__(self, config, num_sensors_per_type):
        super(AdvancedPredictionModel, self).__init__()
        self.config = config
        self.num_sensors_per_type = num_sensors_per_type
        
        # 输入特征维度（所有传感器）
        total_sensors = sum(num_sensors_per_type.values())
        
        # 1. CNN开关（No-CNN消融）
        self.use_cnn = config.CNN_CONFIG.get('use_cnn', True)
        if self.use_cnn:
            self.multiscale_cnn = MultiScaleCNN(
                input_size=total_sensors,
                kernel_sizes=config.CNN_CONFIG['kernel_sizes'],
                num_filters=config.CNN_CONFIG['num_filters'],
                dropout=config.HYPERPARAMETERS['dropout']
            )
            cnn_output_dim = sum(config.CNN_CONFIG['num_filters'])
        else:
            self.multiscale_cnn = None
            cnn_output_dim = total_sensors
        
        # 2. GNN（保留创新点；per-sensor readout → 输出 N*H 维）
        self.use_gnn = config.GNN_CONFIG.get('use_gnn', False)
        if self.use_gnn:
            self.gnn = SensorGNN(
                num_sensors=total_sensors,
                hidden_dim=config.GNN_CONFIG['hidden_dim'],
                num_layers=config.GNN_CONFIG['num_layers'],
                use_position=config.GNN_CONFIG['use_position'],
                num_anchor=config.GNN_CONFIG.get('num_anchor_sensors', total_sensors // 2)
            )
            gnn_output_dim = config.GNN_CONFIG['hidden_dim'] * total_sensors
        else:
            gnn_output_dim = 0
        
        # 3. 双向LSTM输入维度
        if self.use_gnn:
            lstm_input_dim = cnn_output_dim + gnn_output_dim
        else:
            lstm_input_dim = cnn_output_dim
        
        # 双向开关（No-BiDir消融：False=单向LSTM，True=双向LSTM）
        self.bidirectional_flag = config.HYPERPARAMETERS.get('bidirectional', True)
        self.bilstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=config.HYPERPARAMETERS['hidden_size'],
            num_layers=config.HYPERPARAMETERS['num_layers'],
            batch_first=True,
            dropout=config.HYPERPARAMETERS['dropout'] if config.HYPERPARAMETERS['num_layers'] > 1 else 0,
            bidirectional=self.bidirectional_flag
        )

        lstm_output_dim = config.HYPERPARAMETERS['hidden_size'] * (2 if self.bidirectional_flag else 1)

        # 4. 自注意力开关（No-Attention消融：False=关闭，True=启用）
        self.use_attention = config.ATTENTION_CONFIG.get('use_attention', True)
        if self.use_attention:
            self.attention = SelfAttention(
                d_model=lstm_output_dim,
                num_heads=config.ATTENTION_CONFIG['num_heads'],
                dropout=config.HYPERPARAMETERS['dropout']
            )
        
        # 5. 输出层（单步预测；无离层）— 对称 3 层 MLP,两类传感器结构一致
        h = config.HYPERPARAMETERS['hidden_size']
        drop = config.HYPERPARAMETERS['dropout']
        def _make_head(out_dim):
            return nn.Sequential(
                nn.Linear(lstm_output_dim, h * 2),
                nn.ReLU(),
                nn.Dropout(drop),
                nn.Linear(h * 2, h),
                nn.ReLU(),
                nn.Dropout(drop),
                nn.Linear(h, out_dim),
            )
        self.output_layers = nn.ModuleDict({
            'next': nn.ModuleDict({
                '锚杆': _make_head(num_sensors_per_type['锚杆']),
                '围岩': _make_head(num_sensors_per_type['围岩']),
            })
        })

        # 6. 持续锚定 (residual learning): 模型预测残差 Δ̂ = y[t+H] − y[t]
        # 最终输出 ŷ = y[t] + Δ̂。把每个 head 的最末 Linear 零初始化,
        # 训练起点 = persistence,无法比 persistence 更差。
        self.use_persistence_anchor = config.HYPERPARAMETERS.get('use_persistence_anchor', True)
        if self.use_persistence_anchor:
            for st in ('锚杆', '围岩'):
                last_linear = self.output_layers['next'][st][-1]
                nn.init.zeros_(last_linear.weight)
                nn.init.zeros_(last_linear.bias)

        # 记录传感器索引,forward 时切出 persistence 锚点
        # 约定:0 ~ n_anchor-1 = 锚杆;n_anchor ~ total-1 = 围岩
        self._n_anchor = num_sensors_per_type['锚杆']
        self._n_rock = num_sensors_per_type['围岩']
    
    def forward(self, x):
        # x: (batch, seq_len, total_sensors)
        batch_size, seq_len, _ = x.size()
        
        # 1. CNN特征（可关闭）
        if self.use_cnn:
            cnn_out = self.multiscale_cnn(x)  # (batch, seq_len, cnn_output_dim)
        else:
            cnn_out = x  # No-CNN: 直接使用原始输入
        
        # 2. GNN
        if self.use_gnn:
            gnn_out = self.gnn(x)  # (batch, seq_len, hidden_dim)
            lstm_input = torch.cat([cnn_out, gnn_out], dim=-1)
        else:
            lstm_input = cnn_out
        
        # 3. 双向LSTM
        lstm_out, _ = self.bilstm(lstm_input)

        # 4. 自注意力（可关闭）
        if self.use_attention:
            attn_out, attn_weights = self.attention(lstm_out)
            last_hidden = attn_out[:, -1, :]
        else:
            last_hidden = lstm_out[:, -1, :]
            attn_weights = None
        
        # 6. 预测:残差头 + persistence 锚定(可关闭)
        predictions = {'next': {}}
        if self.use_persistence_anchor:
            # 输入 x 的最后一帧即 y[t](标准化空间)
            pers_anchor = x[:, -1, :]                          # (B, total_sensors)
            pers_per_type = {
                '锚杆': pers_anchor[:, :self._n_anchor],
                '围岩': pers_anchor[:, self._n_anchor:self._n_anchor + self._n_rock],
            }
            for sensor_type in ('锚杆', '围岩'):
                delta = self.output_layers['next'][sensor_type](last_hidden)
                predictions['next'][sensor_type] = pers_per_type[sensor_type] + delta
        else:
            for sensor_type in ('锚杆', '围岩'):
                predictions['next'][sensor_type] = self.output_layers['next'][sensor_type](last_hidden)

        return predictions, attn_weights