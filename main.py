# main.py
import os
import argparse
import torch
import numpy as np
import pandas as pd

from config import Config
from models.advanced_models import AdvancedPredictionModel
from utils.data_utils import load_and_prepare_data_advanced, MultiTimeStepDataset, set_seed
from modules.training import AdvancedTrainer
from modules.prediction import AdvancedPredictor, RollingPredictor
from modules.evaluation import ModelEvaluator
from utils.visualization import (
    plot_loss_curves,
    plot_predictions_comparison,
    plot_metrics_dashboard,
    plot_last_sample_prediction
)
import torch.utils.data as Data

# 7天对应的日期标签
DAY_DATES = {
    1: '2025-07-01',
    2: '2025-07-02',
    3: '2025-07-03',
    4: '2025-07-04',
    5: '2025-07-05',
    6: '2025-07-06',
    7: '2025-07-07',
}


def build_loaders(data_dict, batch_size):
    train_dataset = MultiTimeStepDataset(data_dict['train_X'], data_dict['train_y'])
    val_dataset   = MultiTimeStepDataset(data_dict['val_X'],   data_dict['val_y'])
    test_dataset  = MultiTimeStepDataset(data_dict['test_X'],  data_dict['test_y'])

    nw = min(4, os.cpu_count() or 1)
    kw = dict(num_workers=nw, pin_memory=True, persistent_workers=(nw > 0))
    train_loader = Data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  **kw)
    val_loader   = Data.DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, **kw)
    test_loader  = Data.DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, **kw)
    return train_loader, val_loader, test_loader


def run_one_dataset(config, day_idx, seed=42):
    """单个数据集的完整 训练→评估 流程，返回 evaluation_results。"""
    set_seed(seed)

    # 为本次 day 设置独立输出路径
    config.OUTPUT_PATHS = {
        'models':  f'outputs/models/day{day_idx}/',
        'results': f'outputs/results/day{day_idx}/',
        'logs':    f'outputs/logs/day{day_idx}/',
        'plots':   f'outputs/plots/day{day_idx}/',
    }
    for p in config.OUTPUT_PATHS.values():
        os.makedirs(p, exist_ok=True)

    # 加载当天数据
    print(f"\n  📦 加载数据 (day={day_idx})...")
    try:
        data_dict = load_and_prepare_data_advanced(config, day_index=day_idx)
    except Exception as e:
        print(f"  ❌ 数据加载失败: {e}")
        import traceback; traceback.print_exc()
        return None

    train_loader, val_loader, test_loader = build_loaders(
        data_dict, config.HYPERPARAMETERS['batch_size']
    )

    # 创建模型
    model = AdvancedPredictionModel(config, data_dict['num_sensors_per_type'])
    model = model.to(config.DEVICE)
    print(f"  🧠 模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 训练
    print(f"  🚀 开始训练...")
    trainer = AdvancedTrainer(config, model, train_loader, val_loader)
    model, history = trainer.train()

    plot_loss_curves(
        history,
        save_path=os.path.join(config.OUTPUT_PATHS['plots'], 'loss_curves.png')
    )

    # 预测 & 评估
    print(f"  🔮 预测测试集...")
    predictor = AdvancedPredictor(config, model, data_dict['scalers'])
    test_predictions, test_actuals = predictor.predict_batch(test_loader)

    evaluator = ModelEvaluator(config, data_dict['scalers'])
    evaluation_results = evaluator.evaluate_multi_timestep(test_predictions, test_actuals)

    # 保存当天报告
    evaluator.generate_report(
        evaluation_results,
        save_path=os.path.join(config.OUTPUT_PATHS['results'], 'model_metrics.xlsx')
    )
    evaluator.compare_predictions(
        test_predictions, test_actuals,
        save_path=os.path.join(config.OUTPUT_PATHS['results'], 'prediction_comparison.xlsx')
    )
    plot_predictions_comparison(
        test_predictions, test_actuals, config,
        save_path=os.path.join(config.OUTPUT_PATHS['plots'], 'prediction_comparison.png')
    )
    plot_metrics_dashboard(
        evaluation_results,
        save_path=os.path.join(config.OUTPUT_PATHS['plots'], 'metrics_dashboard.png')
    )

    return evaluation_results


def save_summary_excel(all_results, save_path):
    """将 7 个数据集的指标汇总成一张对比表并保存为 Excel。

    表格结构：
        行  = 传感器类型_指标（如 锚杆_MAE, 围岩_R2 ...）
        列  = Dataset-1 ~ Dataset-7
    """
    sensor_types = ['锚杆', '围岩']
    metrics_order = ['MAE', 'RMSE', 'MAPE', 'R2']
    metric_labels = {'MAE': 'MAE', 'RMSE': 'RMSE', 'MAPE': 'MAPE (%)', 'R2': 'R²'}

    # 构建行索引
    row_keys = []
    for st in sensor_types:
        for m in metrics_order:
            row_keys.append((st, m))

    rows = []
    for st, m in row_keys:
        row = {
            '传感器类型': st,
            '指标': metric_labels[m],
        }
        for ds_name, results in all_results.items():
            try:
                val = results['next'][st][m]
                row[ds_name] = round(float(val), 6) if np.isfinite(val) else np.nan
            except (KeyError, TypeError):
                row[ds_name] = np.nan
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.set_index(['传感器类型', '指标'])

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='汇总对比')

    print(f"\n✅ 汇总对比表已保存: {save_path}")
    print(df.to_string())


def main():
    parser = argparse.ArgumentParser(description='顶板预警系统')
    parser.add_argument('--mode', type=str,
                        choices=['train', 'predict', 'full', 'all_days'],
                        default='all_days', help='运行模式')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    args = parser.parse_args()

    Config.create_directories()
    config = Config()

    print("=" * 80)
    print("🏭 顶板预警系统 - 多尺度CNN+双向LSTM+自注意力+GNN（单步预测）")
    print("=" * 80)
    print(f"💻 使用设备: {config.DEVICE}")
    print(f"🎯 运行模式: {args.mode}")
    print("=" * 80)

    # ── 7个数据集循环模式 ──────────────────────────────────────────────
    if args.mode == 'all_days':
        all_results = {}

        for day_idx in range(1, 8):
            date_str = DAY_DATES[day_idx]
            ds_name  = f'Dataset-{day_idx}'
            print(f"\n{'='*80}")
            print(f"  {ds_name}  ({date_str})   [{day_idx}/7]")
            print(f"{'='*80}")

            results = run_one_dataset(config, day_idx, seed=args.seed)
            if results is not None:
                all_results[ds_name] = results
            else:
                print(f"  ⚠️ {ds_name} 运行失败，跳过")

        # 输出汇总表
        if all_results:
            save_summary_excel(
                all_results,
                save_path='outputs/results/all_datasets_summary.xlsx'
            )

        print("\n" + "=" * 80)
        print("🎉 全部数据集训练完成！")
        print("=" * 80)
        print("📁 生成的文件：")
        print("  outputs/results/all_datasets_summary.xlsx  ← 论文对比表")
        print("  outputs/results/day{1~7}/model_metrics.xlsx ← 各天详细指标")
        print("  outputs/plots/day{1~7}/                     ← 各天图表")
        print("=" * 80)
        return

    # ── 原有 full / train / predict 模式（兼容保留）─────────────────────
    set_seed(args.seed)

    print("\n📦 加载数据（全量拼接模式）...")
    try:
        data_dict = load_and_prepare_data_advanced(config)
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        import traceback; traceback.print_exc()
        return

    train_loader, val_loader, test_loader = build_loaders(
        data_dict, config.HYPERPARAMETERS['batch_size']
    )

    print("\n🧠 创建模型...")
    model = AdvancedPredictionModel(config, data_dict['num_sensors_per_type'])
    model = model.to(config.DEVICE)
    print(f"   模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   传感器数量: {data_dict['num_sensors_per_type']}")
    print(f"   GNN状态: {'启用' if config.GNN_CONFIG['use_gnn'] else '禁用'}")

    if args.mode in ['train', 'full']:
        print("\n🚀 开始训练...")
        trainer = AdvancedTrainer(config, model, train_loader, val_loader)
        model, history = trainer.train()
        plot_loss_curves(
            history,
            save_path=os.path.join(config.OUTPUT_PATHS['plots'], 'loss_curves.png')
        )

    if args.mode in ['predict', 'full']:
        if args.mode == 'predict':
            checkpoint_path = os.path.join(config.OUTPUT_PATHS['models'], 'best_model.pth')
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"✅ 已加载模型: {checkpoint_path}")
            else:
                print(f"❌ 模型文件不存在: {checkpoint_path}")
                return

        print("\n🔮 开始预测...")
        predictor = AdvancedPredictor(config, model, data_dict['scalers'])

        last_prediction = predictor.predict_last_training_sample(
            data_dict['train_X'], data_dict['train_y']
        )
        plot_last_sample_prediction(
            last_prediction,
            save_path=os.path.join(config.OUTPUT_PATHS['plots'], 'last_sample_prediction.png')
        )

        print("\n📊 批量预测（测试集）...")
        test_predictions, test_actuals = predictor.predict_batch(test_loader)

        print("\n📈 评估模型...")
        evaluator = ModelEvaluator(config, data_dict['scalers'])
        evaluation_results = evaluator.evaluate_multi_timestep(test_predictions, test_actuals)

        evaluator.generate_report(
            evaluation_results,
            save_path=os.path.join(config.OUTPUT_PATHS['results'], 'model_metrics.xlsx')
        )
        evaluator.compare_predictions(
            test_predictions, test_actuals,
            save_path=os.path.join(config.OUTPUT_PATHS['results'], 'prediction_comparison.xlsx')
        )
        plot_predictions_comparison(
            test_predictions, test_actuals, config,
            save_path=os.path.join(config.OUTPUT_PATHS['plots'], 'prediction_comparison.png')
        )
        plot_metrics_dashboard(
            evaluation_results,
            save_path=os.path.join(config.OUTPUT_PATHS['plots'], 'metrics_dashboard.png')
        )

        rolling_predictor = RollingPredictor(
            model=model,
            scalers=data_dict['scalers'],
            seq_len=config.HYPERPARAMETERS['seq_len'],
            sensor_type_indices=data_dict['sensor_type_indices'],
            device=config.DEVICE
        )
        initial_sequence = data_dict['test_X'][0]
        rolling_predictions = rolling_predictor.rolling_predict(
            initial_sequence=initial_sequence,
            n_steps=10,
            target_sensor_type='围岩'
        )
        print(f"  滚动预测完成: {len(rolling_predictions)} 步")

    print("\n" + "=" * 80)
    print("🎉 程序运行完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
