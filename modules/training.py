# modules/training.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import time
from tqdm import tqdm


def _split_head_last(model: nn.Module):
    """识别"head 末层 Linear"(即预测残差 Δ̂ 的、zero-init 的最后一个 Linear)的参数,
    其余参数归到 rest。

    覆盖两种结构:
      - AdvancedPredictionModel: model.output_layers['next'][stype][-1]
      - 基线 (_PersistenceAnchorMixin + _OutputHead): model.head.heads[stype][-1]

    返回:
      head_last_ids   — set[id]  (用于 set membership 测试,但实际我们直接返回 param 列表)
      head_last_params — list[Tensor]
      rest_params      — list[Tensor]

    如果没找到任何 head 末层 (比如 use_persistence_anchor=False 的特殊情况),
    head_last_params 是空,所有 params 落到 rest_params。
    """
    head_last_modules: list[nn.Linear] = []

    # Full model
    if hasattr(model, 'output_layers') and isinstance(model.output_layers, nn.ModuleDict):
        for sub in model.output_layers.values():
            if isinstance(sub, nn.ModuleDict):
                for seq in sub.values():
                    if hasattr(seq, '__getitem__'):
                        last = seq[-1]
                        if isinstance(last, nn.Linear):
                            head_last_modules.append(last)

    # 基线模型 (_OutputHead.heads[stype])
    if hasattr(model, 'head') and hasattr(model.head, 'heads'):
        for seq in model.head.heads.values():
            if hasattr(seq, '__getitem__'):
                last = seq[-1]
                if isinstance(last, nn.Linear):
                    head_last_modules.append(last)

    head_last_param_ids = set()
    head_last_params = []
    for mod in head_last_modules:
        for p in mod.parameters():
            if p.requires_grad and id(p) not in head_last_param_ids:
                head_last_param_ids.add(id(p))
                head_last_params.append(p)

    rest_params = [p for p in model.parameters()
                   if p.requires_grad and id(p) not in head_last_param_ids]

    return head_last_param_ids, head_last_params, rest_params


class AdvancedTrainer:
    """高级训练器 - 单步预测"""

    def __init__(self, config, model, train_loader, val_loader):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = config.DEVICE

        self.base_lr = config.HYPERPARAMETERS['learning_rate']
        self.warmup_epochs = int(config.HYPERPARAMETERS.get('warmup_epochs', 0))
        self.steps_per_epoch = max(1, len(train_loader))
        self.warmup_total_steps = self.warmup_epochs * self.steps_per_epoch
        self.global_step = 0

        # warmup 启用时，初始 lr 从一个极小值起步；否则直接用 base_lr
        init_lr = self.base_lr / self.warmup_total_steps if self.warmup_total_steps > 0 else self.base_lr
        base_wd = float(config.HYPERPARAMETERS['weight_decay'])
        head_wd_cfg = config.HYPERPARAMETERS.get('head_last_weight_decay', None)

        if head_wd_cfg is not None and float(head_wd_cfg) != base_wd:
            head_wd = float(head_wd_cfg)
            head_last_ids, head_last_params, rest_params = _split_head_last(model)
            print(f"[trainer] head 末层 param 分组: 主组 wd={base_wd}, "
                  f"head_last(zero-init) wd={head_wd}, head_last 张量数={len(head_last_params)}")
            param_groups = [
                {'params': rest_params,      'weight_decay': base_wd},
                {'params': head_last_params, 'weight_decay': head_wd},
            ]
        else:
            head_wd = base_wd
            param_groups = [{'params': list(model.parameters()), 'weight_decay': base_wd}]

        self.optimizer = optim.Adam(param_groups, lr=init_lr)

        lt = config.HYPERPARAMETERS.get('loss_type', 'mse').lower()
        if lt == 'huber':
            self.criterion = nn.SmoothL1Loss(
                beta=config.HYPERPARAMETERS.get('huber_delta', 1.0)
            )
        else:
            self.criterion = nn.MSELoss()

        # 等权重（无离层任务）
        self.sensor_weights = {
            '锚杆': 1.0,
            '围岩': 1.0
        }

        # ReduceLROnPlateau 调度
        lr_cfg = config.HYPERPARAMETERS.get('lr_schedule', {})
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=lr_cfg.get('factor', 0.5),
            patience=lr_cfg.get('patience', 8),
            min_lr=lr_cfg.get('min_lr', 1e-5)
        )

        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }

        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.patience_counter = 0

    def _batch_y_to_device(self, batch_y):
        batch_y_device = {}
        if 'next' in batch_y:
            batch_y_device['next'] = {}
            for sensor_type in batch_y['next'].keys():
                if batch_y['next'][sensor_type] is not None and len(batch_y['next'][sensor_type]) > 0:
                    batch_y_device['next'][sensor_type] = batch_y['next'][sensor_type].to(self.device)
        return batch_y_device

    def _batch_loss_tensor(self, predictions, batch_y_device):
        """单 batch 损失：与「全元素平均 MSE」一致（不再对传感器类型先平均再优化，避免口径扭曲）。"""
        if 'next' not in predictions or 'next' not in batch_y_device:
            return None
        lt = self.config.HYPERPARAMETERS.get('loss_type', 'mse').lower()
        weighted_se = None
        weighted_n = 0.0

        for sensor_type, pred in predictions['next'].items():
            if sensor_type not in batch_y_device['next']:
                continue
            true = batch_y_device['next'][sensor_type]
            if true is None or true.numel() == 0:
                continue
            if len(pred.shape) == 1:
                pred = pred.unsqueeze(0)
            if len(true.shape) == 1:
                true = true.unsqueeze(0)
            if pred.shape[0] != true.shape[0]:
                continue
            w = float(self.sensor_weights.get(sensor_type, 1.0))
            if w == 0.0:
                continue

            if lt == 'huber':
                beta = float(self.config.HYPERPARAMETERS.get('huber_delta', 1.0))
                el = F.smooth_l1_loss(pred, true, beta=beta, reduction='none')
                s = (w * el).sum()
                n = w * pred.numel()
            else:
                diff = pred - true
                s = w * (diff * diff).sum()
                n = w * pred.numel()

            weighted_se = s if weighted_se is None else weighted_se + s
            weighted_n += n

        if weighted_se is None or weighted_n <= 0:
            return None
        return weighted_se / weighted_n

    def _batch_numden_float(self, predictions, batch_y_device):
        """与 _batch_loss_tensor 同定义，返回 (分子, 分母) 便于跨 batch 合并为全局平均。"""
        if 'next' not in predictions or 'next' not in batch_y_device:
            return 0.0, 0.0
        lt = self.config.HYPERPARAMETERS.get('loss_type', 'mse').lower()
        beta = float(self.config.HYPERPARAMETERS.get('huber_delta', 1.0))
        num = 0.0
        den = 0.0
        for sensor_type, pred in predictions['next'].items():
            if sensor_type not in batch_y_device['next']:
                continue
            true = batch_y_device['next'][sensor_type]
            if true is None or true.numel() == 0:
                continue
            if len(pred.shape) == 1:
                pred = pred.unsqueeze(0)
            if len(true.shape) == 1:
                true = true.unsqueeze(0)
            if pred.shape[0] != true.shape[0]:
                continue
            w = float(self.sensor_weights.get(sensor_type, 1.0))
            if w == 0.0:
                continue
            if lt == 'huber':
                el = F.smooth_l1_loss(pred, true, beta=beta, reduction='none')
                num += (w * el).sum().item()
            else:
                diff = (pred - true).detach()
                num += w * (diff * diff).sum().item()
            den += w * pred.numel()
        return num, den

    def _mean_loss_loader(self, loader):
        """eval() 下全局损失 = Σ batch 分子 / Σ batch 分母（与优化目标一致）。"""
        self.model.eval()
        sum_num = 0.0
        sum_den = 0.0
        with torch.no_grad():
            for batch_X, batch_y in tqdm(loader, desc='评估损失', leave=False):
                batch_X = batch_X.to(self.device)
                batch_y_device = self._batch_y_to_device(batch_y)
                predictions, _ = self.model(batch_X)
                n, d = self._batch_numden_float(predictions, batch_y_device)
                sum_num += n
                sum_den += d

        return sum_num / sum_den if sum_den > 0 else 0.0

    def _warmup_step(self):
        """Linear warmup: 在前 warmup_total_steps 步内,lr 从 base_lr/total 线性升到 base_lr。
        warmup 完成后什么也不做,让 ReduceLROnPlateau 在 epoch 末接管。"""
        if self.warmup_total_steps <= 0:
            return
        if self.global_step < self.warmup_total_steps:
            lr = self.base_lr * (self.global_step + 1) / self.warmup_total_steps
            for pg in self.optimizer.param_groups:
                pg['lr'] = lr
        elif self.global_step == self.warmup_total_steps:
            for pg in self.optimizer.param_groups:
                pg['lr'] = self.base_lr

    def train_epoch(self):
        self.model.train()
        pbar = tqdm(self.train_loader, desc='训练中', leave=False)
        for batch_X, batch_y in pbar:
            batch_X = batch_X.to(self.device)
            batch_y_device = self._batch_y_to_device(batch_y)

            self.optimizer.zero_grad()
            predictions, _ = self.model(batch_X)
            loss = self._batch_loss_tensor(predictions, batch_y_device)

            if loss is not None:
                self._warmup_step()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.HYPERPARAMETERS['gradient_clip']
                )
                self.optimizer.step()
                self.global_step += 1
                pbar.set_postfix({'loss': f'{loss.item():.6f}'})

    def train(self):
        print("=" * 80)
        lt = self.config.HYPERPARAMETERS.get('loss_type', 'mse').lower()
        loss_name = 'Huber (SmoothL1)' if lt == 'huber' else 'MSE'
        print(f"🚀 开始训练（{loss_name} + ReduceLROnPlateau；记录曲线用 eval 口径 train/val）")
        print("=" * 80)
        print(f"设备: {self.device}")
        print(f"训练集大小: {len(self.train_loader)} batches")
        print(f"验证集大小: {len(self.val_loader)} batches")
        print(f"模型参数量: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"传感器权重: {self.sensor_weights}")
        print(f"初始学习率: {self.config.HYPERPARAMETERS['learning_rate']}")
        print("=" * 80)

        start_time = time.time()

        for epoch in range(self.config.HYPERPARAMETERS['epochs']):
            epoch_start = time.time()

            # 优化一步（train + dropout）；曲线记录用 eval 下全量 train/val，避免 train/val 口径不一致
            self.train_epoch()
            train_loss = self._mean_loss_loader(self.train_loader)
            val_loss = self._mean_loss_loader(self.val_loader)

            # warmup 阶段不让 plateau 介入,避免 lr 被两边互相覆盖
            if self.global_step >= self.warmup_total_steps:
                self.scheduler.step(val_loss)

            current_lr = self.optimizer.param_groups[0]['lr']

            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rate'].append(current_lr)

            # 保存最佳模型 + 每轮落盘完整 history（供论文 loss 曲线读取 latest_checkpoint）
            improved = val_loss < self.best_val_loss
            if improved:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            self.save_checkpoint(epoch, val_loss, is_best=improved)

            # 打印进度
            epoch_time = time.time() - epoch_start
            elapsed_time = time.time() - start_time
            estimated_total = elapsed_time / (epoch + 1) * self.config.HYPERPARAMETERS['epochs']
            remaining_time = estimated_total - elapsed_time

            if (epoch + 1) % 10 == 0 or epoch < 10:
                print(
                    f"Epoch [{epoch+1}/{self.config.HYPERPARAMETERS['epochs']}] "
                    f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, "
                    f"LR: {current_lr:.6f}, Time: {epoch_time:.2f}s, "
                    f"剩余: {remaining_time/60:.1f}min"
                )

            # 早停
            if self.patience_counter >= self.config.HYPERPARAMETERS['patience']:
                print(f"\n⏹️ 早停于 epoch {epoch+1}")
                break

        # 加载最佳模型
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"\n✅ 已加载最佳模型 (验证损失: {self.best_val_loss:.6f})")

        total_time = time.time() - start_time
        print(f"\n✅ 训练完成！总耗时: {total_time/60:.2f} 分钟 ({total_time/3600:.2f} 小时)")
        print(f"   最佳验证损失: {self.best_val_loss:.6f}")

        return self.model, self.history

    def save_checkpoint(self, epoch, val_loss, is_best=False):
        """保存检查点：latest 每轮更新；best 仅在验证损失刷新时额外写一份。"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'history': self.history
        }

        checkpoint_path = os.path.join(
            self.config.OUTPUT_PATHS['models'],
            'latest_checkpoint.pth'
        )
        torch.save(checkpoint, checkpoint_path)

        if is_best:
            best_path = os.path.join(
                self.config.OUTPUT_PATHS['models'],
                'best_model.pth'
            )
            torch.save(checkpoint, best_path)

    def load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', self.history)
        self.best_val_loss = checkpoint.get('val_loss', float('inf'))
        return checkpoint['epoch']