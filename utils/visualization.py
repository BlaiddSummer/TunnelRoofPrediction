# utils/visualization.py
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False
import seaborn as sns
import numpy as np
import os

SENSOR_TYPES = ['锚杆', '围岩']


def plot_loss_curves(history, save_path=None):
    """绘制loss曲线（原始+平滑，提升可读性）"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    epochs = np.arange(1, len(history['train_loss']) + 1)

    train_loss = np.array(history['train_loss'], dtype=float)
    val_loss = np.array(history['val_loss'], dtype=float)
    lr_hist = np.array(history['learning_rate'], dtype=float)

    # EMA平滑
    def ema(x, alpha=0.25):
        if len(x) == 0:
            return x
        y = np.zeros_like(x, dtype=float)
        y[0] = x[0]
        for i in range(1, len(x)):
            y[i] = alpha * x[i] + (1 - alpha) * y[i - 1]
        return y

    train_smooth = ema(train_loss, alpha=0.25)
    val_smooth = ema(val_loss, alpha=0.25)

    # 左图：loss（原始虚线 + 平滑实线）
    axes[0].plot(epochs, train_loss, 'b--', alpha=0.35, linewidth=1.2, label='训练损失(原始)')
    axes[0].plot(epochs, val_loss, 'r--', alpha=0.35, linewidth=1.2, label='验证损失(原始)')
    axes[0].plot(epochs, train_smooth, 'b-', linewidth=2.2, label='训练损失(平滑)')
    axes[0].plot(epochs, val_smooth, 'r-', linewidth=2.2, label='验证损失(平滑)')

    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('训练和验证损失曲线（含平滑）', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # 右图：学习率
    axes[1].plot(epochs, lr_hist, 'g-', label='学习率', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Learning Rate', fontsize=12)
    axes[1].set_title('学习率变化曲线', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale('log')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Loss曲线已保存至: {save_path}")

    plt.close()


def plot_predictions_comparison(predictions, actuals, config, save_path=None):
    """绘制预测对比图（折线版：真实值 vs 预测值）"""
    num_timesteps = len(config.PREDICTION_TIMESTEPS)
    num_sensor_types = len(SENSOR_TYPES)

    fig, axes = plt.subplots(num_timesteps, num_sensor_types, figsize=(12, 5.5 * num_timesteps))

    if num_timesteps == 1:
        axes = axes.reshape(1, -1)
    if num_sensor_types == 1:
        axes = axes.reshape(-1, 1)

    sensor_types = SENSOR_TYPES
    timestep_names = list(config.PREDICTION_TIMESTEPS.keys())

    def to_series(arr):
        """
        转为每样本一个值：
        - (N,) -> (N,)
        - (N,S) -> 对S取均值 -> (N,)
        """
        arr = np.asarray(arr)
        if arr.ndim == 1:
            return arr
        if arr.ndim >= 2:
            return arr.reshape(arr.shape[0], -1).mean(axis=1)
        return arr

    def moving_avg(x, w=5):
        if len(x) < w:
            return x
        kernel = np.ones(w) / w
        return np.convolve(x, kernel, mode='same')

    for i, time_name in enumerate(timestep_names):
        for j, sensor_type in enumerate(sensor_types):
            ax = axes[i, j]

            if (time_name in predictions and sensor_type in predictions[time_name] and
                time_name in actuals and sensor_type in actuals[time_name]):

                pred = to_series(predictions[time_name][sensor_type])
                true = to_series(actuals[time_name][sensor_type])

                n = min(len(pred), len(true))
                if n == 0:
                    ax.text(0.5, 0.5, '无数据', ha='center', va='center',
                            transform=ax.transAxes, fontsize=14)
                    ax.set_title(f'{time_name} - {sensor_type}', fontsize=12)
                    continue

                pred = pred[:n]
                true = true[:n]

                # 为避免太密，最多显示前300点
                n_show = min(300, n)
                x = np.arange(n_show)

                pred_show = pred[:n_show]
                true_show = true[:n_show]

                # 可视化平滑（仅用于画图，不影响实际误差）
                pred_plot = moving_avg(pred_show, w=5)
                true_plot = moving_avg(true_show, w=5)

                ax.plot(x, true_plot, label='真实值', linewidth=2.0, alpha=0.95, color='#1f77b4')
                ax.plot(x, pred_plot, label='预测值', linewidth=2.0, alpha=0.95, color='#ff7f0e')

                mae_local = float(np.mean(np.abs(pred_show - true_show)))
                rmse_local = float(np.sqrt(np.mean((pred_show - true_show) ** 2)))
                ax.text(
                    0.02, 0.95,
                    f'MAE={mae_local:.4f}  RMSE={rmse_local:.4f}',
                    transform=ax.transAxes,
                    va='top',
                    fontsize=9,
                    bbox=dict(facecolor='white', alpha=0.75, edgecolor='none')
                )

                ax.set_xlabel('样本索引', fontsize=10)
                ax.set_ylabel('值', fontsize=10)
                ax.set_title(f'{time_name} - {sensor_type}（折线对比）', fontsize=12, fontweight='bold')
                ax.legend(fontsize=9, loc='best')
                ax.grid(True, alpha=0.25)

            else:
                ax.text(0.5, 0.5, '无数据', ha='center', va='center',
                        transform=ax.transAxes, fontsize=14)
                ax.set_title(f'{time_name} - {sensor_type}', fontsize=12)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 预测对比图（折线版）已保存至: {save_path}")

    plt.close()


def plot_metrics_dashboard(evaluation_results, save_path=None):
    """绘制指标仪表盘"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 准备数据
    metrics_data = []
    for time_name in evaluation_results.keys():
        for sensor_type in evaluation_results[time_name].keys():
            metrics = evaluation_results[time_name][sensor_type]
            metrics_data.append({
                '时间点': time_name,
                '传感器类型': sensor_type,
                'MAE': metrics.get('MAE', np.nan),
                'RMSE': metrics.get('RMSE', np.nan),
                'MAPE': metrics.get('MAPE', np.nan),
                'R2': metrics.get('R2', np.nan)
            })

    if not metrics_data:
        print("⚠️ 没有评估数据可绘制")
        return

    import pandas as pd
    df = pd.DataFrame(metrics_data)

    # 1. MAE对比
    ax1 = axes[0, 0]
    pivot_mae = df.pivot(index='传感器类型', columns='时间点', values='MAE')
    pivot_mae.plot(kind='bar', ax=ax1, width=0.8, color=['#4ECDC4', '#FF6B6B'])
    ax1.set_title('MAE对比', fontsize=14, fontweight='bold')
    ax1.set_ylabel('MAE', fontsize=12)
    ax1.legend(title='时间点', fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.tick_params(axis='x', rotation=0)

    # 2. RMSE对比
    ax2 = axes[0, 1]
    pivot_rmse = df.pivot(index='传感器类型', columns='时间点', values='RMSE')
    pivot_rmse.plot(kind='bar', ax=ax2, width=0.8, color=['#FF6B6B', '#4ECDC4'])
    ax2.set_title('RMSE对比', fontsize=14, fontweight='bold')
    ax2.set_ylabel('RMSE', fontsize=12)
    ax2.legend(title='时间点', fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.tick_params(axis='x', rotation=0)

    # 3. MAPE对比
    ax3 = axes[1, 0]
    pivot_mape = df.pivot(index='传感器类型', columns='时间点', values='MAPE')
    pivot_mape.plot(kind='bar', ax=ax3, width=0.8, color=['#96CEB4', '#FFEAA7'])
    ax3.set_title('MAPE对比 (%)', fontsize=14, fontweight='bold')
    ax3.set_ylabel('MAPE (%)', fontsize=12)
    ax3.legend(title='时间点', fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.tick_params(axis='x', rotation=0)

    # 4. R²对比
    ax4 = axes[1, 1]
    pivot_r2 = df.pivot(index='传感器类型', columns='时间点', values='R2')
    pivot_r2.plot(kind='bar', ax=ax4, width=0.8, color=['#FFB347', '#87CEEB'])
    ax4.set_title('R²对比', fontsize=14, fontweight='bold')
    ax4.set_ylabel('R²', fontsize=12)
    ax4.legend(title='时间点', fontsize=10)
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.tick_params(axis='x', rotation=0)
    ax4.axhline(y=0, color='r', linestyle='--', linewidth=1)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 指标仪表盘已保存至: {save_path}")

    plt.close()


def plot_last_sample_prediction(prediction_result, save_path=None):
    """绘制训练集最后一组数据的预测结果"""
    predictions = prediction_result['predictions']
    actuals = prediction_result['actuals']

    num_timesteps = len(predictions)
    num_sensor_types = len(SENSOR_TYPES)

    fig, axes = plt.subplots(num_timesteps, num_sensor_types, figsize=(12, 6 * num_timesteps))

    if num_timesteps == 1:
        axes = axes.reshape(1, -1)
    if num_sensor_types == 1:
        axes = axes.reshape(-1, 1)

    sensor_types = SENSOR_TYPES
    timestep_names = list(predictions.keys())

    for i, time_name in enumerate(timestep_names):
        for j, sensor_type in enumerate(sensor_types):
            ax = axes[i, j]

            if (sensor_type in predictions[time_name] and
                sensor_type in actuals.get(time_name, {})):

                pred = predictions[time_name][sensor_type]
                true = actuals[time_name][sensor_type]

                # 绘制对比
                n_sensors = len(pred)
                x = np.arange(n_sensors)
                width = 0.35

                ax.bar(x - width / 2, true, width, label='真实值', alpha=0.8)
                ax.bar(x + width / 2, pred, width, label='预测值', alpha=0.8)

                ax.set_xlabel('传感器索引', fontsize=10)
                ax.set_ylabel('值', fontsize=10)
                ax.set_title(f'{time_name} - {sensor_type}', fontsize=12, fontweight='bold')
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3, axis='y')
                ax.set_xticks(x)
            else:
                ax.text(0.5, 0.5, '无数据', ha='center', va='center',
                        transform=ax.transAxes, fontsize=14)
                ax.set_title(f'{time_name} - {sensor_type}', fontsize=12)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 最后一组数据预测图已保存至: {save_path}")

    plt.close()