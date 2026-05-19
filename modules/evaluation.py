# modules/evaluation.py
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

# 与 data_utils / prediction 一致：无离层任务
SENSOR_TYPES = ['锚杆', '围岩']


class ModelEvaluator:
    """模型评估器（统一尺度版：四个指标在同一空间计算）"""

    def __init__(self, config, scalers):
        self.config = config
        self.scalers = scalers

    def calculate_mape(self, y_true, y_pred):
        """计算MAPE（平均绝对百分比误差）"""
        mask = y_true != 0
        if mask.sum() == 0:
            return np.nan
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    def inverse_transform(self, values, sensor_type):
        """反标准化数据"""
        if sensor_type not in self.scalers:
            return values

        scaler = self.scalers[sensor_type]["scaler"]
        columns = self.scalers[sensor_type]["columns"]

        # 处理不同维度的数据
        if len(values.shape) == 1:
            # 1D数组：单个样本或多个样本的单个传感器
            if len(values) == len(columns):
                # 单个样本，多个传感器
                values_2d = values.reshape(1, -1)
                denormalized = scaler.inverse_transform(values_2d)
                return denormalized[0]
            else:
                # 多个样本，单个传感器
                return values
        elif len(values.shape) == 2:
            # 2D数组：(samples, sensors)
            if values.shape[1] == len(columns):
                return scaler.inverse_transform(values)
            else:
                return values
        else:
            return values

    def calculate_metrics(self, y_true, y_pred, sensor_type=None, metric_space="denorm"):
        """统一尺度计算评估指标

        Args:
            y_true: 真实值（通常是标准化后的）
            y_pred: 预测值（通常是标准化后的）
            sensor_type: 传感器类型（用于反标准化）
            metric_space: 'denorm' 或 'norm'
                - denorm: MAE/RMSE/MAPE/R2 全在反标准化后计算（默认，常用于报告）
                - norm:   MAE/RMSE/MAPE/R2 全在标准化后计算（常用于模型内部比较）
        """
        # 展平数组
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()

        # 移除NaN和Inf
        mask = np.isfinite(y_true_flat) & np.isfinite(y_pred_flat)
        y_true_clean = y_true_flat[mask]
        y_pred_clean = y_pred_flat[mask]

        if len(y_true_clean) == 0:
            return {
                "MAE": np.nan,
                "RMSE": np.nan,
                "MAPE": np.nan,
                "R2": np.nan,
            }

        # 默认在标准化空间评估
        y_true_eval = y_true_clean
        y_pred_eval = y_pred_clean

        # 若指定反标准化空间，则四个指标统一在反标准化后计算
        if metric_space == "denorm" and sensor_type is not None and sensor_type in self.scalers:
            try:
                n_sensors = len(self.scalers[sensor_type]["columns"])
                if n_sensors > 0 and (len(y_true_clean) % n_sensors == 0):
                    y_true_reshaped = y_true_clean.reshape(-1, n_sensors)
                    y_pred_reshaped = y_pred_clean.reshape(-1, n_sensors)

                    y_true_eval = self.inverse_transform(y_true_reshaped, sensor_type).flatten()
                    y_pred_eval = self.inverse_transform(y_pred_reshaped, sensor_type).flatten()
                else:
                    print(
                        f"  ⚠️ {sensor_type}: 无法按 n_sensors={n_sensors} reshape，"
                        "改用标准化空间计算全部指标"
                    )
            except Exception as e:
                print(f"  ⚠️ {sensor_type}: 反标准化失败，改用标准化空间计算全部指标: {e}")

        # 反标准化后再做一次安全清洗
        mask_eval = np.isfinite(y_true_eval) & np.isfinite(y_pred_eval)
        y_true_eval = y_true_eval[mask_eval]
        y_pred_eval = y_pred_eval[mask_eval]

        if len(y_true_eval) == 0:
            return {
                "MAE": np.nan,
                "RMSE": np.nan,
                "MAPE": np.nan,
                "R2": np.nan,
            }

        mae = mean_absolute_error(y_true_eval, y_pred_eval)
        rmse = np.sqrt(mean_squared_error(y_true_eval, y_pred_eval))
        mape = self.calculate_mape(y_true_eval, y_pred_eval)

        # 常数序列下R²数学上不可定义，返回NaN更稳妥
        if np.allclose(np.var(y_true_eval), 0.0):
            r2 = np.nan
        else:
            r2 = r2_score(y_true_eval, y_pred_eval)

        return {
            "MAE": mae,
            "RMSE": rmse,
            "MAPE": mape,
            "R2": r2,
        }

    def evaluate_multi_timestep(self, predictions, actuals, metric_space="denorm"):
        """评估多时间步预测（四个指标在同一尺度计算）"""
        results = {}

        for time_name in self.config.PREDICTION_TIMESTEPS.keys():
            results[time_name] = {}

            for sensor_type in SENSOR_TYPES:
                if (
                    time_name in predictions
                    and sensor_type in predictions[time_name]
                    and time_name in actuals
                    and sensor_type in actuals[time_name]
                ):
                    pred = predictions[time_name][sensor_type]
                    true = actuals[time_name][sensor_type]

                    # 确保维度匹配
                    if pred.shape != true.shape:
                        min_samples = min(len(pred), len(true))
                        pred = pred[:min_samples]
                        true = true[:min_samples]

                    metrics = self.calculate_metrics(
                        true,
                        pred,
                        sensor_type=sensor_type,
                        metric_space=metric_space,
                    )
                    results[time_name][sensor_type] = metrics

        return results

    def generate_report(self, evaluation_results, save_path=None, metric_space="denorm"):
        """生成评估报告"""
        report_data = []

        for time_name in evaluation_results.keys():
            for sensor_type in evaluation_results[time_name].keys():
                metrics = evaluation_results[time_name][sensor_type]
                report_data.append(
                    {
                        "预测时间点": time_name,
                        "传感器类型": sensor_type,
                        "MAE": metrics.get("MAE", np.nan),
                        "RMSE": metrics.get("RMSE", np.nan),
                        "MAPE (%)": metrics.get("MAPE", np.nan),
                        "R²": metrics.get("R2", np.nan),
                    }
                )

        df = pd.DataFrame(report_data)

        # 打印报告
        print("\n" + "=" * 80)
        if metric_space == "denorm":
            print("📊 模型评估报告（MAE/RMSE/MAPE/R² 均在反标准化数据上计算）")
        else:
            print("📊 模型评估报告（MAE/RMSE/MAPE/R² 均在标准化数据上计算）")
        print("=" * 80)
        print(df.to_string(index=False))
        print("=" * 80)

        # 保存报告
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            df.to_excel(save_path, index=False)
            print(f"\n✅ 报告已保存至: {save_path}")

        return df

    def compare_predictions(self, predictions, actuals, save_path=None):
        """对比预测值与真实值"""
        comparison_data = []

        for time_name in self.config.PREDICTION_TIMESTEPS.keys():
            for sensor_type in SENSOR_TYPES:
                if (
                    time_name in predictions
                    and sensor_type in predictions[time_name]
                    and time_name in actuals
                    and sensor_type in actuals[time_name]
                ):
                    pred = predictions[time_name][sensor_type]
                    true = actuals[time_name][sensor_type]

                    # 确保维度匹配
                    if pred.shape != true.shape:
                        min_samples = min(len(pred), len(true))
                        pred = pred[:min_samples]
                        true = true[:min_samples]

                    # 反标准化（用于显示）
                    if sensor_type in self.scalers:
                        try:
                            pred_denorm = self.inverse_transform(pred, sensor_type)
                            true_denorm = self.inverse_transform(true, sensor_type)
                        except Exception:
                            pred_denorm = pred
                            true_denorm = true
                    else:
                        pred_denorm = pred
                        true_denorm = true

                    # 为每个传感器创建对比数据
                    if len(pred_denorm.shape) == 1:
                        # 单个传感器
                        for i in range(len(pred_denorm)):
                            comparison_data.append(
                                {
                                    "预测时间点": time_name,
                                    "传感器类型": sensor_type,
                                    "传感器索引": i,
                                    "预测值": pred_denorm[i],
                                    "真实值": true_denorm[i],
                                    "误差": abs(pred_denorm[i] - true_denorm[i]),
                                    "相对误差 (%)": (
                                        abs((pred_denorm[i] - true_denorm[i]) / true_denorm[i] * 100)
                                        if true_denorm[i] != 0
                                        else np.nan
                                    ),
                                }
                            )
                    else:
                        # 多个传感器
                        for i in range(len(pred_denorm)):
                            for j in range(pred_denorm.shape[1]):
                                comparison_data.append(
                                    {
                                        "预测时间点": time_name,
                                        "传感器类型": sensor_type,
                                        "样本索引": i,
                                        "传感器索引": j,
                                        "预测值": pred_denorm[i, j],
                                        "真实值": true_denorm[i, j],
                                        "误差": abs(pred_denorm[i, j] - true_denorm[i, j]),
                                        "相对误差 (%)": (
                                            abs((pred_denorm[i, j] - true_denorm[i, j]) / true_denorm[i, j] * 100)
                                            if true_denorm[i, j] != 0
                                            else np.nan
                                        ),
                                    }
                                )

        df = pd.DataFrame(comparison_data)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            df.to_excel(save_path, index=False)
            print(f"✅ 对比数据已保存至: {save_path}")

        return df