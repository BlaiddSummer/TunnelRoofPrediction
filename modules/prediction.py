# modules/prediction.py
import torch
import numpy as np
from tqdm import tqdm

# 与 data_utils.SENSOR_TYPES 一致：无离层任务
SENSOR_TYPES = ['锚杆', '围岩']


class AdvancedPredictor:
    """高级预测器 - 单步预测 + 滚动预测（参考论文方法）"""
    
    def __init__(self, config, model, scalers):
        self.config = config
        self.model = model
        self.scalers = scalers
        self.device = config.DEVICE
        self.model.eval()
    
    def predict_single(self, X):
        """单样本预测（单步预测）"""
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).unsqueeze(0).to(self.device)
            predictions, attn_weights = self.model(X_tensor)
            
            # 转换为numpy（单步预测：只有'next'）
            pred_dict = {}
            if 'next' in predictions:
                pred_dict['next'] = {}
                for sensor_type in predictions['next'].keys():
                    pred_dict['next'][sensor_type] = predictions['next'][sensor_type].cpu().numpy()[0]
            
            return pred_dict, attn_weights
    
    def predict_batch(self, data_loader):
        """批量预测（单步预测）"""
        self.model.eval()
        all_predictions = []
        all_actuals = []
        
        with torch.no_grad():
            for batch_X, batch_y in tqdm(data_loader, desc='预测中'):
                batch_X = batch_X.to(self.device)
                
                predictions, _ = self.model(batch_X)
                
                # 转换为numpy（单步预测：只有'next'）
                batch_pred_dict = {}
                if 'next' in predictions:
                    batch_pred_dict['next'] = {}
                    for sensor_type in predictions['next'].keys():
                        batch_pred_dict['next'][sensor_type] = predictions['next'][sensor_type].cpu().numpy()
                
                # 转换实际值（单步预测：只有'next'）
                batch_actual_dict = {}
                if 'next' in batch_y:
                    batch_actual_dict['next'] = {}
                    for sensor_type in batch_y['next'].keys():
                        if batch_y['next'][sensor_type] is not None:
                            batch_actual_dict['next'][sensor_type] = batch_y['next'][sensor_type].numpy()
                
                all_predictions.append(batch_pred_dict)
                all_actuals.append(batch_actual_dict)
        
        # 合并所有批次（单步预测：只有'next'）
        merged_predictions = {}
        merged_actuals = {}
        
        merged_predictions['next'] = {}
        merged_actuals['next'] = {}
        for sensor_type in SENSOR_TYPES:
            pred_list = []
            actual_list = []
            
            for batch_pred, batch_actual in zip(all_predictions, all_actuals):
                if 'next' in batch_pred and sensor_type in batch_pred['next']:
                    pred_list.append(batch_pred['next'][sensor_type])
                if 'next' in batch_actual and sensor_type in batch_actual['next']:
                    if batch_actual['next'][sensor_type] is not None:
                        actual_list.append(batch_actual['next'][sensor_type])
            
            if pred_list:
                merged_predictions['next'][sensor_type] = np.concatenate(pred_list, axis=0)
            if actual_list:
                merged_actuals['next'][sensor_type] = np.concatenate(actual_list, axis=0)
        
        return merged_predictions, merged_actuals
    
    def predict_last_training_sample(self, train_X, train_y):
        """预测训练集最后一组数据（单步预测）"""
        print("\n" + "="*80)
        print("🔮 预测训练集最后一组数据（单步预测）")
        print("="*80)
        
        # 获取最后一个样本
        last_X = train_X[-1:]  # 保持2D形状
        last_y = {}
        if 'next' in train_y:
            last_y['next'] = {}
            for sensor_type in train_y['next'].keys():
                if len(train_y['next'][sensor_type]) > 0:
                    last_y['next'][sensor_type] = train_y['next'][sensor_type][-1:]
        
        # 进行预测
        predictions, attn_weights = self.predict_single(last_X[0])
        
        # 反标准化预测值
        denormalized_predictions = {}
        if 'next' in predictions:
            denormalized_predictions['next'] = {}
            for sensor_type in predictions['next'].keys():
                if sensor_type in self.scalers:
                    scaler = self.scalers[sensor_type]['scaler']
                    columns = self.scalers[sensor_type]['columns']
                    pred_values = predictions['next'][sensor_type]
                    
                    # 反标准化
                    if len(pred_values) == len(columns):
                        pred_2d = pred_values.reshape(1, -1)
                        denormalized = scaler.inverse_transform(pred_2d)
                        denormalized_predictions['next'][sensor_type] = denormalized[0]
                    else:
                        denormalized_predictions['next'][sensor_type] = pred_values
                else:
                    denormalized_predictions['next'][sensor_type] = predictions['next'][sensor_type]
        
        # 反标准化真实值
        denormalized_actuals = {}
        if 'next' in last_y:
            denormalized_actuals['next'] = {}
            for sensor_type in last_y['next'].keys():
                if len(last_y['next'][sensor_type]) > 0:
                    actual_values = last_y['next'][sensor_type][0]
                    if sensor_type in self.scalers:
                        scaler = self.scalers[sensor_type]['scaler']
                        columns = self.scalers[sensor_type]['columns']
                        
                        if len(actual_values) == len(columns):
                            actual_2d = actual_values.reshape(1, -1)
                            denormalized = scaler.inverse_transform(actual_2d)
                            denormalized_actuals['next'][sensor_type] = denormalized[0]
                        else:
                            denormalized_actuals['next'][sensor_type] = actual_values
                    else:
                        denormalized_actuals['next'][sensor_type] = actual_values
        
        print("\n预测结果（反标准化后）:")
        if 'next' in denormalized_predictions:
            print(f"\n  下一个时间步:")
            for sensor_type in SENSOR_TYPES:
                if sensor_type in denormalized_predictions['next']:
                    pred = denormalized_predictions['next'][sensor_type]
                    actual = denormalized_actuals.get('next', {}).get(sensor_type, None)
                    
                    print(f"    {sensor_type}:")
                    print(f"      预测值: {pred}")
                    if actual is not None:
                        print(f"      真实值: {actual}")
                        error = np.abs(pred - actual)
                        print(f"      误差: {error}")
        
        return {
            'predictions': denormalized_predictions,
            'actuals': denormalized_actuals,
            'predictions_normalized': predictions,
            'actuals_normalized': last_y
        }


class RollingPredictor:
    """滚动预测器 - 实现论文中的滚动预测策略（支持GNN）"""
    
    def __init__(self, model, scalers, seq_len, sensor_type_indices, device):
        """
        初始化滚动预测器
        
        Args:
            model: 训练好的模型
            scalers: 标准化器字典
            seq_len: 输入序列长度
            sensor_type_indices: 传感器类型索引映射
            device: 设备
        """
        self.model = model
        self.scalers = scalers
        self.seq_len = seq_len
        self.sensor_type_indices = sensor_type_indices
        self.device = device
        self.model.eval()
    
    def rolling_predict(self, initial_sequence, n_steps, target_sensor_type='围岩'):
        """
        滚动预测n步（论文方法：预测值反馈到输入序列）
        
        Args:
            initial_sequence: 初始输入序列 (seq_len, total_features)
            n_steps: 要预测的步数
            target_sensor_type: 目标传感器类型（用于反馈）
        
        Returns:
            predictions: 预测值列表
        """
        predictions = []
        current_sequence = initial_sequence.copy()
        
        self.model.eval()
        with torch.no_grad():
            for step in range(n_steps):
                # 准备输入
                input_seq = torch.FloatTensor(current_sequence).unsqueeze(0).to(self.device)
                
                # 预测（单步预测：只有'next'）
                pred_dict, _ = self.model(input_seq)
                
                if 'next' not in pred_dict or target_sensor_type not in pred_dict['next']:
                    break
                
                pred_value = pred_dict['next'][target_sensor_type].cpu().numpy()[0]
                predictions.append(pred_value)
                
                # 滚动：将预测值反馈到输入序列（论文方法）
                # 移除最早的时间步，添加新的时间步
                new_row = current_sequence[-1].copy()
                
                # 找到目标传感器类型的索引位置
                if target_sensor_type in self.sensor_type_indices:
                    start_idx, end_idx = self.sensor_type_indices[target_sensor_type]
                    # 用预测值替换目标传感器类型的值
                    if len(pred_value) == (end_idx - start_idx):
                        new_row[start_idx:end_idx] = pred_value
                
                # 更新序列：移除第一行，添加新行
                current_sequence = np.vstack([
                    current_sequence[1:],  # 移除第一行
                    new_row.reshape(1, -1)  # 添加新行
                ])
        
        return predictions
    
    def rolling_predict_with_feedback(self, initial_sequence, n_steps, 
                                      actual_values=None, feedback_interval=4):
        """
        滚动预测n步，带误差反馈（论文方法：n+4时反馈实际值）
        
        Args:
            initial_sequence: 初始输入序列
            n_steps: 要预测的步数
            actual_values: 实际值列表（用于反馈）
            feedback_interval: 反馈间隔（论文中为4）
        
        Returns:
            predictions: 预测值列表
        """
        predictions = []
        current_sequence = initial_sequence.copy()
        
        self.model.eval()
        with torch.no_grad():
            for step in range(n_steps):
                # 准备输入
                input_seq = torch.FloatTensor(current_sequence).unsqueeze(0).to(self.device)
                
                # 预测
                pred_dict, _ = self.model(input_seq)
                
                # 获取预测值（这里简化处理，预测所有传感器类型）
                pred_values = {}
                if 'next' in pred_dict:
                    for sensor_type in pred_dict['next'].keys():
                        pred_values[sensor_type] = pred_dict['next'][sensor_type].cpu().numpy()[0]
                
                predictions.append(pred_values)
                
                # 滚动：将预测值或实际值反馈到输入序列
                new_row = current_sequence[-1].copy()
                
                # 如果到了反馈间隔，使用实际值（论文方法）
                if actual_values is not None and (step + 1) % feedback_interval == 0:
                    if step < len(actual_values):
                        # 使用实际值
                        for sensor_type, actual_val in actual_values[step].items():
                            if sensor_type in self.sensor_type_indices:
                                start_idx, end_idx = self.sensor_type_indices[sensor_type]
                                if len(actual_val) == (end_idx - start_idx):
                                    new_row[start_idx:end_idx] = actual_val
                else:
                    # 使用预测值
                    for sensor_type, pred_val in pred_values.items():
                        if sensor_type in self.sensor_type_indices:
                            start_idx, end_idx = self.sensor_type_indices[sensor_type]
                            if len(pred_val) == (end_idx - start_idx):
                                new_row[start_idx:end_idx] = pred_val
                
                # 更新序列
                current_sequence = np.vstack([
                    current_sequence[1:],
                    new_row.reshape(1, -1)
                ])
        
        return predictions