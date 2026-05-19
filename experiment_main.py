# experiment_main.py
"""
对比实验 + 消融实验 统一入口
===============================
实验组：
  对比实验（4个基线）: LSTM / BiLSTM / CNN-LSTM / Transformer
  消融实验（5个变体）: Full_Model / w/o_CNN / w/o_GNN / w/o_Attention / w/o_BiDir

运行：
    python experiment_main.py                          # 跑全部实验 (单种子)
    python experiment_main.py --exps LSTM,Full_Model   # 只跑指定实验
    python experiment_main.py --days 1,2,3             # 只跑指定天
    python experiment_main.py --skip_existing          # 跳过已有结果

    # 多种子模式 (mean ± std)：
    python experiment_main.py --days 1 --seeds 42,123,456,789,2024
    # 输出落到 outputs/experiments_seeds/seed{S}/，与单种子目录隔离

输出：
    outputs/experiments/<exp_name>/day<N>/                 各模型各天的详细结果 (单种子)
    outputs/experiments/experiment_summary.xlsx            汇总对比表
    outputs/experiments_seeds/seed<S>/<exp_name>/day<N>/   多种子模式产物
    outputs/experiments_seeds/seed<S>/experiment_summary.xlsx
"""
import os
import copy
import argparse
import numpy as np
import pandas as pd
import torch
import torch.utils.data as Data

from config.config import Config
from models.advanced_models import AdvancedPredictionModel
from models.baseline_models import LSTMModel, BiLSTMModel, CNNLSTMModel, TransformerModel
from models.ml_baselines import ML_BASELINE_REGISTRY
from utils.data_utils import load_and_prepare_data_advanced, MultiTimeStepDataset, set_seed
from modules.training import AdvancedTrainer
from modules.prediction import AdvancedPredictor
from modules.ml_training import MLTrainer
from modules.ml_prediction import MLPredictor
from modules.evaluation import ModelEvaluator

# ── 实验注册表 ─────────────────────────────────────────────────────────────────
#   type='advanced' : 使用 AdvancedPredictionModel，通过 overrides 做消融
#   type='baseline' : 使用独立基线模型类
# ─────────────────────────────────────────────────────────────────────────────
# Advanced 系列共享的正则强度：Full Model + 全部 ablation 参数远多于基线,
# 需要比基线 (dropout=0.3, wd=5e-4) 更强的正则才公平。
# 一视同仁应用到所有 advanced 实验,确保消融对比的控制变量。
ADVANCED_REG = {
    'HYPERPARAMETERS.dropout': 0.4,
    'HYPERPARAMETERS.weight_decay': 0.001,
}

EXPERIMENTS = {
    # ---------- 对比实验:深度学习基线 (DL) ----------
    'LSTM':        {'type': 'baseline', 'model_cls': LSTMModel},
    'BiLSTM':      {'type': 'baseline', 'model_cls': BiLSTMModel},
    'CNN-LSTM':    {'type': 'baseline', 'model_cls': CNNLSTMModel},
    # Transformer 已从默认 baseline 移除 (同类论文罕见, 也比 Full 强反而抢戏)。
    # 仍可通过 --exps Transformer 单独跑;默认 run-all 不含它。
    'Transformer': {'type': 'baseline', 'model_cls': TransformerModel},
    # ---------- 对比实验:经典 ML 基线 ----------
    'RF':       {'type': 'ml', 'model_cls': ML_BASELINE_REGISTRY['RF']},
    'XGBoost':  {'type': 'ml', 'model_cls': ML_BASELINE_REGISTRY['XGBoost']},
    'SVR':      {'type': 'ml', 'model_cls': ML_BASELINE_REGISTRY['SVR']},
    'MLP':      {'type': 'ml', 'model_cls': ML_BASELINE_REGISTRY['MLP']},
    # ---------- 消融实验 ----------
    'Full_Model':    {'type': 'advanced', 'overrides': {**ADVANCED_REG}},
    'w/o_CNN':       {'type': 'advanced', 'overrides': {**ADVANCED_REG, 'CNN_CONFIG.use_cnn': False}},
    'w/o_GNN':       {'type': 'advanced', 'overrides': {**ADVANCED_REG, 'GNN_CONFIG.use_gnn': False}},
    'w/o_Attention': {'type': 'advanced', 'overrides': {**ADVANCED_REG, 'ATTENTION_CONFIG.use_attention': False}},
    'w/o_BiDir':     {'type': 'advanced', 'overrides': {**ADVANCED_REG, 'HYPERPARAMETERS.bidirectional': False}},
}

# 不在默认全跑里的 (旧 baseline, 故意保留但不自动 run)
EXCLUDE_FROM_RUN_ALL = {'Transformer'}

DAY_DATES = {
    1: '2025-07-01', 2: '2025-07-02', 3: '2025-07-03',
    4: '2025-07-04', 5: '2025-07-05', 6: '2025-07-06', 7: '2025-07-07',
}

SENSOR_TYPES = ['锚杆', '围岩']
METRICS      = ['MAE', 'RMSE', 'MAPE', 'R2']


# ── 工具函数 ───────────────────────────────────────────────────────────────────

def make_config(overrides: dict = None) -> Config:
    """
    创建隔离的 Config 实例（各实验互不干扰）。
    overrides 格式: {'CNN_CONFIG.use_cnn': False, 'HYPERPARAMETERS.bidirectional': False}
    """
    config = Config.__new__(Config)
    # 深拷贝所有字典属性，避免修改类级别共享对象
    dict_attrs = [
        'HYPERPARAMETERS', 'CNN_CONFIG', 'ATTENTION_CONFIG',
        'GNN_CONFIG', 'DATA_CONFIG', 'OUTPUT_PATHS',
        'EVALUATION_CONFIG', 'PREDICTION_TIMESTEPS',
    ]
    for attr in dict_attrs:
        if hasattr(Config, attr):
            setattr(config, attr, copy.deepcopy(getattr(Config, attr)))
    config.DEVICE = Config.DEVICE  # torch.device 不可变，无需拷贝

    if overrides:
        for dotted_key, value in overrides.items():
            parts = dotted_key.split('.', 1)
            attr_name = parts[0]
            if len(parts) == 2:
                d = getattr(config, attr_name)
                if isinstance(d, dict):
                    d[parts[1]] = value
            else:
                setattr(config, attr_name, value)

    return config


def create_model(exp_cfg: dict, config: Config,
                 num_sensors_per_type: dict,
                 sensor_type_indices: dict | None = None,
                 seed: int = 42):
    """根据实验配置创建对应模型。返回类型:
       advanced/baseline -> torch.nn.Module
       ml                -> _MLBaselineBase (鸭子类型,有 fit/predict/state_dict)
    """
    if exp_cfg['type'] == 'advanced':
        return AdvancedPredictionModel(config, num_sensors_per_type)
    if exp_cfg['type'] == 'ml':
        cls = exp_cfg['model_cls']
        # ML baseline 构造签名: (config, num_sensors_per_type, sensor_type_indices, seed)
        return cls(config, num_sensors_per_type,
                   sensor_type_indices=sensor_type_indices, seed=seed)
    # baseline 深度学习
    cls = exp_cfg['model_cls']
    return cls(config, num_sensors_per_type)


def build_loaders(data_dict: dict, batch_size: int):
    train_ds = MultiTimeStepDataset(data_dict['train_X'], data_dict['train_y'])
    val_ds   = MultiTimeStepDataset(data_dict['val_X'],   data_dict['val_y'])
    test_ds  = MultiTimeStepDataset(data_dict['test_X'],  data_dict['test_y'])
    nw = min(4, os.cpu_count() or 1)
    kw = dict(num_workers=nw, pin_memory=True, persistent_workers=(nw > 0))
    return (
        Data.DataLoader(train_ds, batch_size=batch_size, shuffle=True,  **kw),
        Data.DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **kw),
        Data.DataLoader(test_ds,  batch_size=batch_size, shuffle=False, **kw),
    )


# ── 单次实验运行 ───────────────────────────────────────────────────────────────

def run_one(exp_name: str, exp_cfg: dict, day_idx: int,
            seed: int = 42, skip_existing: bool = False,
            output_root: str = 'outputs/experiments') -> dict | None:
    """
    运行一个（实验, 天）组合，返回 evaluation_results 或 None（失败时）。
    evaluation_results 格式：{'next': {'锚杆': {'MAE':..,'RMSE':..,'MAPE':..,'R2':..}, '围岩': {...}}}

    output_root: 输出根目录，单种子默认 'outputs/experiments'，
                 多种子模式调用方传入 'outputs/experiments_seeds/seed{S}'。
    """
    result_path = f'{output_root}/{exp_name}/day{day_idx}/results/model_metrics.xlsx'

    # 断点续传：如果已有结果则跳过训练，直接读取指标
    if skip_existing and os.path.exists(result_path):
        print(f"  [跳过] {exp_name} day{day_idx} — 结果已存在")
        try:
            df = pd.read_excel(result_path)
            results = {'next': {}}
            for _, row in df.iterrows():
                st = row['传感器类型']
                results['next'][st] = {
                    'MAE':  row.get('MAE',  np.nan),
                    'RMSE': row.get('RMSE', np.nan),
                    'MAPE': row.get('MAPE (%)', np.nan),
                    'R2':   row.get('R²',   np.nan),
                }
            return results
        except Exception:
            pass   # 读取失败则重新训练

    set_seed(seed)
    config = make_config(exp_cfg.get('overrides', {}))

    # 设置本次实验的独立输出目录
    exp_dir = f'{output_root}/{exp_name}/day{day_idx}'
    config.OUTPUT_PATHS = {
        'models':  f'{exp_dir}/models/',
        'results': f'{exp_dir}/results/',
        'logs':    f'{exp_dir}/logs/',
        'plots':   f'{exp_dir}/plots/',
    }
    for p in config.OUTPUT_PATHS.values():
        os.makedirs(p, exist_ok=True)

    # 加载数据
    print(f"    加载数据 (day={day_idx})...")
    try:
        data_dict = load_and_prepare_data_advanced(config, day_index=day_idx)
    except Exception as e:
        print(f"    数据加载失败: {e}")
        return None

    train_loader, val_loader, test_loader = build_loaders(
        data_dict, config.HYPERPARAMETERS['batch_size']
    )

    # 创建模型 (ML baseline 需要 sensor_type_indices + seed 接入 persistence anchor)
    model = create_model(
        exp_cfg, config, data_dict['num_sensors_per_type'],
        sensor_type_indices=data_dict.get('sensor_type_indices'),
        seed=seed,
    )
    model = model.to(config.DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    if exp_cfg['type'] == 'ml':
        print(f"    模型类型: ML baseline ({exp_cfg['model_cls'].__name__})")
    else:
        print(f"    模型参数量: {n_params:,}")

    # 训练 + 预测: NN 走 AdvancedTrainer / AdvancedPredictor;
    #              ML 走 MLTrainer / MLPredictor (接口同形)。
    if exp_cfg['type'] == 'ml':
        trainer = MLTrainer(config, model, train_loader, val_loader)
        model, _ = trainer.train()
        predictor = MLPredictor(config, model, data_dict['scalers'])
    else:
        trainer = AdvancedTrainer(config, model, train_loader, val_loader)
        model, _ = trainer.train()
        predictor = AdvancedPredictor(config, model, data_dict['scalers'])
    test_preds, test_actuals = predictor.predict_batch(test_loader)

    evaluator = ModelEvaluator(config, data_dict['scalers'])
    results = evaluator.evaluate_multi_timestep(test_preds, test_actuals)

    # 保存当次结果
    evaluator.generate_report(
        results,
        save_path=os.path.join(config.OUTPUT_PATHS['results'], 'model_metrics.xlsx')
    )
    return results


# ── 汇总输出 ───────────────────────────────────────────────────────────────────

def save_experiment_summary(all_results: dict, save_path: str):
    """
    all_results 格式：{exp_name: {day_idx: results_or_None}}

    生成两个 Sheet：
      '平均结果(论文对比)' : 各模型在7天上的平均指标，即论文对比表
      '逐天详细结果'       : 全部原始数据（可用于绘图）
    """
    records = []
    for exp_name, day_results in all_results.items():
        for day_idx, res in day_results.items():
            if res is None:
                continue
            row = {'模型': exp_name, '数据集': f'Dataset-{day_idx}',
                   '日期': DAY_DATES.get(day_idx, '')}
            for st in SENSOR_TYPES:
                for m in METRICS:
                    col = f'{st}_{m}'
                    try:
                        val = res['next'][st][m]
                        row[col] = round(float(val), 6) if np.isfinite(val) else np.nan
                    except (KeyError, TypeError):
                        row[col] = np.nan
            records.append(row)

    if not records:
        print("  没有可汇总的结果")
        return

    df_all = pd.DataFrame(records)

    # 计算7天平均
    metric_cols = [f'{st}_{m}' for st in SENSOR_TYPES for m in METRICS]
    avg_records = []
    for exp_name in all_results.keys():
        sub = df_all[df_all['模型'] == exp_name]
        avg_row = {'模型': exp_name}
        for col in metric_cols:
            avg_row[col] = sub[col].mean()
        avg_records.append(avg_row)
    df_avg = pd.DataFrame(avg_records)

    # 列标题更友好
    rename_map = {}
    for st in SENSOR_TYPES:
        for m in METRICS:
            label = m if m != 'R2' else 'R²'
            rename_map[f'{st}_{m}'] = f'{st}_{label}'
    df_avg = df_avg.rename(columns=rename_map)
    df_all = df_all.rename(columns=rename_map)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
        df_avg.to_excel(writer, sheet_name='平均结果(论文对比)', index=False)
        df_all.to_excel(writer, sheet_name='逐天详细结果',       index=False)

    print(f"\n{'='*80}")
    print(f"✅ 实验汇总已保存: {save_path}")
    print(f"{'='*80}")
    print("\n【平均结果（论文对比表）】")
    print(df_avg.to_string(index=False))


# ── 主函数 ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='顶板预警系统 - 对比/消融实验')
    parser.add_argument(
        '--exps', type=str, default='',
        help='逗号分隔的实验名称，留空则运行全部。'
             f'可选: {", ".join(EXPERIMENTS.keys())}'
    )
    parser.add_argument(
        '--days', type=str, default='',
        help='逗号分隔的天索引（1~7），留空则运行全部7天。'
    )
    parser.add_argument('--seed', type=int, default=42, help='随机种子 (单种子模式)')
    parser.add_argument(
        '--seeds', type=str, default='',
        help='逗号分隔的多个随机种子；给了就启用多种子模式 (mean ± std 评估)。'
             '输出到 outputs/experiments_seeds/seed<S>/，与单种子目录隔离。'
    )
    parser.add_argument(
        '--skip_existing', action='store_true',
        help='如果某实验某天的结果文件已存在，跳过训练直接读取'
    )
    args = parser.parse_args()

    # 解析实验列表
    if args.exps:
        exp_names = [e.strip() for e in args.exps.split(',')]
        unknown = [e for e in exp_names if e not in EXPERIMENTS]
        if unknown:
            print(f"未知实验名: {unknown}")
            print(f"可选实验: {list(EXPERIMENTS.keys())}")
            return
    else:
        # 默认 run-all: 不含 EXCLUDE_FROM_RUN_ALL (例如 Transformer)
        exp_names = [e for e in EXPERIMENTS.keys() if e not in EXCLUDE_FROM_RUN_ALL]
        if EXCLUDE_FROM_RUN_ALL:
            print(f"  (默认跑全套时已排除: {sorted(EXCLUDE_FROM_RUN_ALL)};"
                  f" 如需跑请用 --exps Transformer 等显式指定)")

    # 解析天列表
    if args.days:
        day_indices = [int(d.strip()) for d in args.days.split(',')]
    else:
        day_indices = list(range(1, 8))

    # 解析种子列表：单种子 or 多种子
    if args.seeds.strip():
        seed_list = [int(s.strip()) for s in args.seeds.split(',') if s.strip()]
        multi_seed = True  # 显式指定 --seeds 时始终用 experiments_seeds 目录
    else:
        seed_list = [args.seed]
        multi_seed = False

    Config.create_directories()

    print('=' * 80)
    print('🏭 顶板预警系统 - 对比实验 & 消融实验')
    print('=' * 80)
    print(f'实验列表 ({len(exp_names)} 个): {exp_names}')
    print(f'数据天数 ({len(day_indices)} 天): {day_indices}')
    print(f'设备: {Config.DEVICE}')
    print(f'随机种子: {seed_list}{" (多种子模式)" if multi_seed else ""}')
    print(f'总训练次数: {len(seed_list) * len(exp_names) * len(day_indices)}')
    print('=' * 80)

    grand_total = len(seed_list) * len(exp_names) * len(day_indices)
    grand_done = 0

    for seed in seed_list:
        if multi_seed:
            output_root = f'outputs/experiments_seeds/seed{seed}'
            summary_path = f'{output_root}/experiment_summary.xlsx'
        else:
            output_root = 'outputs/experiments'
            summary_path = 'outputs/experiments/experiment_summary.xlsx'

        if multi_seed:
            print(f"\n{'#' * 80}")
            print(f"#  种子 {seed}    输出: {output_root}")
            print(f"{'#' * 80}")

        all_results = {name: {} for name in exp_names}
        sub_total = len(exp_names) * len(day_indices)
        sub_done = 0

        for exp_name in exp_names:
            exp_cfg = EXPERIMENTS[exp_name]
            print(f"\n{'='*80}")
            print(f"  实验: {exp_name}  (type={exp_cfg['type']})  seed={seed}")
            if exp_cfg['type'] == 'advanced':
                print(f"  消融开关: {exp_cfg.get('overrides', {}) or '无（完整模型）'}")
            print(f"{'='*80}")

            for day_idx in day_indices:
                sub_done += 1
                grand_done += 1
                date_str = DAY_DATES.get(day_idx, '')
                progress = (f"[seed {seed}: {sub_done}/{sub_total}]"
                            if multi_seed else f"[{sub_done}/{sub_total}]")
                print(f"\n  {progress}  {exp_name} / Dataset-{day_idx} ({date_str})  "
                      f"(总进度 {grand_done}/{grand_total})")

                results = run_one(
                    exp_name, exp_cfg, day_idx,
                    seed=seed,
                    skip_existing=args.skip_existing,
                    output_root=output_root,
                )

                if results is not None:
                    all_results[exp_name][day_idx] = results
                    for st in SENSOR_TYPES:
                        if st in results.get('next', {}):
                            m = results['next'][st]
                            print(f"    {st}: MAE={m.get('MAE', 'nan'):.4f}  "
                                  f"RMSE={m.get('RMSE', 'nan'):.4f}  "
                                  f"MAPE={m.get('MAPE', 'nan'):.2f}%  "
                                  f"R²={m.get('R2', 'nan'):.4f}")
                else:
                    print(f"  ⚠️ {exp_name} day{day_idx} 失败，跳过")

        # 每个种子结束后保存自己的 summary
        save_experiment_summary(all_results, save_path=summary_path)

    print('\n' + '=' * 80)
    print('🎉 所有实验完成！')
    print('=' * 80)
    print('📁 输出文件：')
    if multi_seed:
        for seed in seed_list:
            print(f'  outputs/experiments_seeds/seed{seed}/experiment_summary.xlsx')
        print(f'\n  ← 多种子聚合 (mean ± std)：')
        print(f'    python scripts/aggregate_multiseed_r2_pers.py '
              f'--seeds_root outputs/experiments_seeds '
              f'--days {",".join(map(str, day_indices))}')
    else:
        print('  outputs/experiments/experiment_summary.xlsx  ← 论文对比表（两个Sheet）')
        print('  outputs/experiments/<实验名>/day<N>/         ← 各实验各天详细结果')
    print('=' * 80)


if __name__ == '__main__':
    main()
