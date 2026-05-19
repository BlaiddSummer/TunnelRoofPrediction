# utils/data_utils.py
import torch
import torch.utils.data as Data
import pandas as pd
import numpy as np
import os
import re
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

# 本项目不包含离层（无离层预测任务）
SENSOR_TYPES = ['锚杆', '围岩']


def set_seed(seed=42):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def identify_sensor_type(df):
    """根据数据内容识别传感器类型"""
    sensor_type_mapping = {
        '锚杆工作阻力': '锚杆',
        '顶板离层位移': '离层',
        '围岩应力': '围岩',
        61: '锚杆',
        20: '离层',
        33: '围岩',
        '007': '锚杆',
        '010': '离层',
        '004': '围岩'
    }

    if 'sensor_name' in df.columns:
        sensor_names = df['sensor_name'].dropna().unique()
        for sensor_name in sensor_names:
            for key, sensor_type in sensor_type_mapping.items():
                if isinstance(key, str) and key in str(sensor_name):
                    return sensor_type

    if 'name_code' in df.columns:
        name_codes = df['name_code'].dropna().unique()
        for name_code in name_codes:
            if name_code in sensor_type_mapping:
                return sensor_type_mapping[name_code]

    if 'sensor_code' in df.columns:
        sensor_codes = df['sensor_code'].dropna().unique()
        for sensor_code in sensor_codes:
            if str(sensor_code) in sensor_type_mapping:
                return sensor_type_mapping[str(sensor_code)]

    return None


def convert_to_numeric(series):
    """将文本类型的数值转换为数字"""
    numeric_series = pd.to_numeric(series, errors='coerce')

    if numeric_series.isna().sum() > len(series) * 0.5:
        cleaned_series = series.astype(str).apply(
            lambda x: re.sub(r'[^\d\.\-]', '', str(x).strip()) if pd.notna(x) else ''
        )
        numeric_series = pd.to_numeric(cleaned_series, errors='coerce')

    if numeric_series.isna().sum() > len(series) * 0.3:
        def extract_number(text):
            if pd.isna(text):
                return np.nan
            text_str = str(text)
            match = re.search(r'-?\d+\.?\d*', text_str)
            if match:
                try:
                    return float(match.group())
                except Exception:
                    return np.nan
            return np.nan
        numeric_series = series.apply(extract_number)

    return numeric_series


def load_sensor_files(data_dir, file_name, sensor_type):
    """加载单个传感器数据，若原文件不存在则自动拼接分日文件。
    分日文件命名规则：SQLA7(1).csv, SQLA7(2).csv, ...
    """
    file_path = os.path.join(data_dir, file_name)

    if os.path.exists(file_path):
        return read_sensor_csv(file_path, expected_type=sensor_type)

    # 尝试查找分日文件
    stem = os.path.splitext(file_name)[0]  # e.g. 'SQLA7'
    pattern = re.compile(r'^' + re.escape(stem) + r'\((\d+)\)\.csv$', re.IGNORECASE)
    try:
        all_files = os.listdir(data_dir)
    except OSError:
        all_files = []

    split_files = sorted(
        [f for f in all_files if pattern.match(f)],
        key=lambda f: int(pattern.match(f).group(1))
    )

    if not split_files:
        print(f"  ⚠️ 文件不存在: {file_name}（也未找到分日文件）")
        return None

    print(f"  📅 自动拼接 {len(split_files)} 个分日文件: {stem}(1)~({len(split_files)})")
    dfs = []
    for sf in split_files:
        df = read_sensor_csv(os.path.join(data_dir, sf), expected_type=sensor_type)
        if df is not None:
            dfs.append(df)

    if not dfs:
        return None

    combined = pd.concat(dfs, ignore_index=True).sort_values('时间').reset_index(drop=True)
    combined['sensor_type'] = sensor_type
    print(f"     {stem} 合并后共 {len(combined)} 条记录")
    return combined


def read_sensor_csv(file_path, expected_type=None):
    """读取单个传感器CSV文件"""
    try:
        for encoding in ['utf-8', 'gbk', 'gb2312', 'latin1']:
            try:
                df = pd.read_csv(file_path, encoding=encoding, low_memory=False)
                break
            except UnicodeDecodeError:
                continue
        else:
            print(f"  ⚠️ 无法读取文件: {os.path.basename(file_path)} (编码问题)")
            return None

        identified_type = identify_sensor_type(df)
        sensor_type = identified_type if identified_type else expected_type

        if not sensor_type:
            print(f"  ⚠️ 无法识别类型: {os.path.basename(file_path)}")
            return None

        if 'date' in df.columns:
            date_col = 'date'
        elif '时间' in df.columns:
            date_col = '时间'
        else:
            print(f"  ⚠️ 未找到时间列: {os.path.basename(file_path)}")
            return None

        if 'value' in df.columns:
            value_col = 'value'
        elif '值' in df.columns:
            value_col = '值'
        else:
            print(f"  ⚠️ 未找到数值列: {os.path.basename(file_path)}")
            return None

        numeric_values = convert_to_numeric(df[value_col])
        valid_count = numeric_values.notna().sum()
        total_count = len(numeric_values)

        if valid_count == 0:
            print(f"  ⚠️ {os.path.basename(file_path)}: 无法转换任何数值，跳过")
            return None

        if valid_count < total_count * 0.1:
            print(f"  ⚠️ {os.path.basename(file_path)}: 只有 {valid_count}/{total_count} 个有效值，可能有问题")

        result_df = pd.DataFrame({
            '时间': pd.to_datetime(df[date_col], errors='coerce'),
            'value': numeric_values
        })
        result_df = result_df.dropna(subset=['时间', 'value']).sort_values('时间').reset_index(drop=True)

        if len(result_df) == 0:
            print(f"  ⚠️ {os.path.basename(file_path)}: 处理后无有效数据，跳过")
            return None

        result_df['sensor_type'] = sensor_type
        print(f"  ✅ {os.path.basename(file_path)}: {len(result_df)}条记录, 类型={sensor_type}, 有效值={valid_count}/{total_count}")
        return result_df

    except Exception as e:
        print(f"  ❌ 读取失败 {os.path.basename(file_path)}: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_and_prepare_data_advanced(config, day_index=None):
    """高级数据加载 - 单步预测（参考论文方法）"""
    print("=" * 80)
    print("[Advanced] Loading data for single-step prediction")
    print("=" * 80)

    data_dir = config.DATA_CONFIG['data_dir']
    file_mapping = config.DATA_CONFIG['file_mapping']
    seq_len = config.HYPERPARAMETERS['seq_len']

    # 预测时长 (从 config 读取). 1 = 单步 (5秒); 60 = 5 分钟; 120 = 10 分钟 ...
    prediction_timestep = int(config.PREDICTION_TIMESTEPS.get('next', 1))

    sensor_type_data = {}
    sensor_column_mapping = {}
    scalers = {}

    for sensor_type, file_list in file_mapping.items():
        print(f"\n处理 {sensor_type} 数据...")
        sensor_dataframes = {}

        for file_name in file_list:
            if day_index is not None:
                stem = os.path.splitext(file_name)[0]
                day_file = f"{stem}({day_index}).csv"
                day_path = os.path.join(data_dir, day_file)
                if not os.path.exists(day_path):
                    print(f"  ⚠️ 文件不存在: {day_file}")
                    continue
                df = read_sensor_csv(day_path, expected_type=sensor_type)
            else:
                df = load_sensor_files(data_dir, file_name, sensor_type)
            if df is not None:
                sensor_name = os.path.splitext(file_name)[0]
                sensor_dataframes[sensor_name] = df[['时间', 'value']].rename(columns={'value': sensor_name})

        if not sensor_dataframes:
            print(f"  ⚠️ 没有成功加载任何 {sensor_type} 数据")
            continue

        combined_df = None
        for sensor_name, df in sensor_dataframes.items():
            if combined_df is None:
                combined_df = df.copy()
            else:
                combined_df = pd.merge_asof(
                    combined_df.sort_values('时间'),
                    df.sort_values('时间'),
                    on='时间',
                    direction='nearest',
                    tolerance=pd.Timedelta(config.DATA_CONFIG['time_tolerance'])
                )

        combined_df = combined_df.sort_values('时间').reset_index(drop=True)
        sensor_columns = [col for col in combined_df.columns if col != '时间']
        sensor_column_mapping[sensor_type] = sensor_columns

        print(f"  ✅ {sensor_type}: {len(sensor_columns)} 个传感器")
        print(f"     时间范围: {combined_df['时间'].min()} 到 {combined_df['时间'].max()}")
        print(
            f"     传感器列名: {sensor_columns[:5]}..." if len(sensor_columns) > 5 else f"     传感器列名: {sensor_columns}"
        )

        print(f"  处理 {sensor_type} 缺失值...")
        valid_sensors = []
        for col in sensor_columns:
            missing_before = combined_df[col].isna().sum()
            total = len(combined_df)
            valid_count = total - missing_before
            valid_ratio = valid_count / total if total > 0 else 0

            if valid_ratio < 0.05:
                print(f"    ⚠️ {col}: 有效数据比例 {valid_ratio * 100:.2f}% ({valid_count}/{total}) < 5%，将被排除")
                continue

            valid_sensors.append(col)
            combined_df[col] = combined_df[col].interpolate(method='linear', limit_direction='both')
            combined_df[col] = combined_df[col].bfill().ffill()
            missing_after = combined_df[col].isna().sum()
            if missing_before > 0:
                print(f"    {col}: 缺失 {missing_before} -> {missing_after}")

        if not valid_sensors:
            print(f"  ⚠️ {sensor_type} 没有有效传感器，跳过")
            continue

        combined_df = combined_df[['时间'] + valid_sensors]
        sensor_column_mapping[sensor_type] = valid_sensors

        combined_df = combined_df.dropna(how='all', subset=valid_sensors)
        combined_df[valid_sensors] = combined_df[valid_sensors].ffill().bfill()
        combined_df = combined_df.dropna(subset=valid_sensors)

        if len(combined_df) == 0:
            print(f"  ⚠️ {sensor_type} 处理后无数据，跳过")
            continue

        scaler = RobustScaler()
        combined_df[valid_sensors] = scaler.fit_transform(combined_df[valid_sensors])
        scalers[sensor_type] = {'scaler': scaler, 'columns': valid_sensors}
        sensor_type_data[sensor_type] = combined_df

        print(f"  ✅ {sensor_type} 处理完成: {len(combined_df)} 个时间点, {len(valid_sensors)} 个传感器")

    if not sensor_type_data:
        raise ValueError("没有成功加载任何数据")

    print(f"\n寻找公共时间范围...")
    all_time_ranges = []
    for sensor_type, df in sensor_type_data.items():
        time_range = (df['时间'].min(), df['时间'].max())
        all_time_ranges.append(time_range)
        print(f"  {sensor_type}: {time_range[0]} 到 {time_range[1]}")

    common_start = max([tr[0] for tr in all_time_ranges])
    common_end = min([tr[1] for tr in all_time_ranges])

    if common_start >= common_end:
        raise ValueError(f"没有公共时间范围！开始时间 {common_start} >= 结束时间 {common_end}")

    print(f"  公共时间范围: {common_start} 到 {common_end}")

    time_range = pd.date_range(start=common_start, end=common_end, freq='5s')
    merged_df = pd.DataFrame({'时间': time_range})
    print(f"  统一时间点数量: {len(time_range)}")

    print(f"\n对齐各类型数据到统一时间索引...")
    for sensor_type, df in sensor_type_data.items():
        sensor_columns = sensor_column_mapping[sensor_type]
        df_filtered = df[(df['时间'] >= common_start) & (df['时间'] <= common_end)].copy()
        if len(df_filtered) == 0:
            print(f"  ⚠️ {sensor_type} 在公共时间范围内无数据")
            continue
        merged_df = pd.merge_asof(
            merged_df.sort_values('时间'),
            df_filtered.sort_values('时间'),
            on='时间',
            direction='nearest',
            tolerance=pd.Timedelta(config.DATA_CONFIG['time_tolerance'])
        )
        print(f"  ✅ {sensor_type}: {len(sensor_columns)} 个传感器已对齐")

    value_columns = [col for col in merged_df.columns if col != '时间']
    print(f"\n处理对齐后的缺失值...")

    for sensor_type, columns in sensor_column_mapping.items():
        valid_columns = [col for col in columns if col in value_columns]
        if not valid_columns:
            continue
        for col in valid_columns:
            missing_before = merged_df[col].isna().sum()
            if missing_before > 0:
                merged_df[col] = merged_df[col].interpolate(method='linear', limit_direction='both')
                merged_df[col] = merged_df[col].bfill().ffill()
                missing_after = merged_df[col].isna().sum()
                if missing_after > 0:
                    print(f"  ⚠️ {col}: 仍有 {missing_after} 个缺失值")

    merged_df = merged_df.dropna(how='all', subset=value_columns)
    merged_df[value_columns] = merged_df[value_columns].ffill().bfill()
    merged_df = merged_df.dropna(subset=value_columns)

    if len(merged_df) == 0:
        raise ValueError("处理后数据为空")

    print(f"✅ 数据对齐完成: {len(merged_df)} 个时间点, {len(value_columns)} 个传感器")

    data = merged_df[value_columns].values

    sensor_type_indices = {}
    current_idx = 0
    for sensor_type, columns in sensor_column_mapping.items():
        valid_columns = [col for col in columns if col in value_columns]
        if valid_columns:
            num_sensors = len(valid_columns)
            sensor_type_indices[sensor_type] = (current_idx, current_idx + num_sensors)
            current_idx += num_sensors

    print(f"\n创建序列数据（前瞻预测）...")
    print(f"  序列长度: {seq_len}")
    print(f"  预测步数: {prediction_timestep} 步 ({prediction_timestep * 5} 秒后)")

    # 前瞻预测：预测 prediction_timestep 步之后的值
    X, y_dict = [], {}
    y_dict['next'] = {}
    for sensor_type in SENSOR_TYPES:
        y_dict['next'][sensor_type] = []

    for i in range(len(data) - seq_len - prediction_timestep + 1):
        X.append(data[i:i + seq_len])
        # 目标：seq_len 之后的第 prediction_timestep 步 (从 1 开始计)
        target_idx = i + seq_len + prediction_timestep - 1
        if target_idx < len(data):
            for sensor_type in SENSOR_TYPES:
                if sensor_type in sensor_type_indices:
                    start_idx, end_idx = sensor_type_indices[sensor_type]
                    target_values = data[target_idx, start_idx:end_idx]
                    y_dict['next'][sensor_type].append(target_values)
                else:
                    y_dict['next'][sensor_type].append(np.array([]))

    X = np.array(X)

    for sensor_type in y_dict['next'].keys():
        if len(y_dict['next'][sensor_type]) > 0:
            y_dict['next'][sensor_type] = np.array(y_dict['next'][sensor_type])
        else:
            y_dict['next'][sensor_type] = np.array([])

    print(f"✅ 序列数据创建完成: {len(X)} 个样本")

    # 划分方式：chronological（默认，与论文主实验一致）| random（打乱样本再按比例切，仅建议用于 Fig.8 等「收敛形态」展示）
    split_mode = config.DATA_CONFIG.get('split_mode', 'chronological')
    if split_mode == 'random':
        seed = int(config.DATA_CONFIG.get('random_split_seed', 42))
        rng = np.random.RandomState(seed)
        perm = rng.permutation(len(X))
        X = X[perm]
        for st in y_dict['next'].keys():
            arr = y_dict['next'][st]
            if arr is not None and len(arr) > 0:
                y_dict['next'][st] = np.asarray(arr)[perm]
        print('\n⚠️  当前使用 **随机样本划分** (split_mode=random, seed=%d)。' % seed)
        print('    验证集与训练集来自同一时间轴上的混合抽样，loss 曲线更易呈下降趋势；')
        print('    **正式实验 / 论文指标** 请使用默认时序划分 (chronological)，勿用此模式。')

    train_size = int(len(X) * (1 - config.DATA_CONFIG['test_size'] - config.DATA_CONFIG['val_size']))
    val_size = int(len(X) * config.DATA_CONFIG['val_size'])

    train_X = X[:train_size]
    val_X = X[train_size:train_size + val_size]
    test_X = X[train_size + val_size:]

    train_y = {'next': {st: y_dict['next'][st][:train_size]
                        for st in y_dict['next'].keys()}}
    val_y = {'next': {st: y_dict['next'][st][train_size:train_size + val_size]
                      for st in y_dict['next'].keys()}}
    test_y = {'next': {st: y_dict['next'][st][train_size + val_size:]
                       for st in y_dict['next'].keys()}}

    # 使用全部训练数据（论文方法）
    sample_ratio = config.DATA_CONFIG.get('sample_ratio', 1.0)
    if sample_ratio < 1.0:
        print(f"\n📊 数据采样: 使用 {sample_ratio * 100:.1f}% 的训练数据")
        sample_strategy = config.DATA_CONFIG.get('sample_strategy', 'uniform')
        original_train_size = len(train_X)
        if sample_strategy == 'uniform':
            sample_indices = np.linspace(0, len(train_X) - 1, int(len(train_X) * sample_ratio), dtype=int)
            train_X = train_X[sample_indices]
            for sensor_type in train_y['next'].keys():
                if len(train_y['next'][sensor_type]) > 0:
                    train_y['next'][sensor_type] = train_y['next'][sensor_type][sample_indices]
        print(f"   采样后训练集: {len(train_X)} 样本（原 {original_train_size} 样本）")
    else:
        print(f"\n📊 使用全部训练数据（论文方法）")

    num_sensors_per_type = {}
    for sensor_type in SENSOR_TYPES:
        if sensor_type in sensor_type_indices:
            start_idx, end_idx = sensor_type_indices[sensor_type]
            num_sensors_per_type[sensor_type] = end_idx - start_idx
        else:
            num_sensors_per_type[sensor_type] = 0

    print(f"\n✅ 数据加载完成！")
    print(f"   训练集: {len(train_X)} 样本")
    print(f"   验证集: {len(val_X)} 样本")
    print(f"   测试集: {len(test_X)} 样本")
    print(f"   传感器数量: {num_sensors_per_type}")

    return {
        'train_X': train_X,
        'train_y': train_y,
        'val_X': val_X,
        'val_y': val_y,
        'test_X': test_X,
        'test_y': test_y,
        'scalers': scalers,
        'sensor_column_mapping': sensor_column_mapping,
        'num_sensors_per_type': num_sensors_per_type,
        'sensor_type_indices': sensor_type_indices,
        'merged_df': merged_df,
        'value_columns': value_columns
    }


class MultiTimeStepDataset(Data.Dataset):
    """多时间步预测数据集（单步预测版）"""
    def __init__(self, X, y_dict):
        self.X = torch.FloatTensor(X)
        self.y_dict = {}
        # 单步预测：只有'next'
        if 'next' in y_dict:
            self.y_dict['next'] = {}
            for sensor_type in y_dict['next'].keys():
                if len(y_dict['next'][sensor_type]) > 0:
                    self.y_dict['next'][sensor_type] = torch.FloatTensor(y_dict['next'][sensor_type])
                else:
                    self.y_dict['next'][sensor_type] = None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        y = {}
        # 单步预测：只有'next'
        if 'next' in self.y_dict:
            y['next'] = {}
            for sensor_type in self.y_dict['next'].keys():
                if self.y_dict['next'][sensor_type] is not None:
                    y['next'][sensor_type] = self.y_dict['next'][sensor_type][idx]
                else:
                    y['next'][sensor_type] = torch.tensor([])
        return self.X[idx], y