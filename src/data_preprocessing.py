import pandas as pd
import numpy as np

def load_data(file_path):
    """
    Load historical demand data from a CSV file.
    
    Parameters:
    file_path (str): The path to the historical demand CSV file.
    
    Returns:
    DataFrame: A pandas DataFrame containing the historical demand data.
    """
    demand_data = pd.read_csv(file_path, encoding='gbk')
    demand_data['过账日期'] = pd.to_datetime(demand_data['过账日期'])
    return demand_data

def remove_duplicates(demand_data):
    """
    去除重复数据，基于过账日期、工厂编码、物料编码完全相同的记录
    
    Args:
        demand_data: 原始数据DataFrame
        
    Returns:
        处理后的DataFrame和重复数据示例
    """
    # 获取重复数据的数量
    duplicates = demand_data[demand_data.duplicated(subset=['过账日期', '工厂编码', '物料编码'], keep=False)]
    duplicate_count = len(duplicates)
    
    # 获取10行重复数据示例
    duplicate_samples = duplicates.head(10)
    
    # 去除重复数据，保留第一次出现的记录
    cleaned_data = demand_data.drop_duplicates(subset=['过账日期', '工厂编码', '物料编码'], keep='first')
    
    print(f"发现 {duplicate_count} 条重复数据")
    print("\n重复数据示例（前10行）：")
    print(duplicate_samples[['过账日期', '工厂编码', '物料编码', '需求量']])
    print(f"\n去重后数据量: {len(cleaned_data)} 条")
    
    return cleaned_data

def handle_negative_demand(demand_data):
    """
    处理负数需求量数据
    
    Args:
        demand_data: 原始数据DataFrame
        
    Returns:
        处理后的DataFrame
    """
    # 找出负数需求量的行
    negative_demand = demand_data[demand_data['需求量'] < 0]
    
    if len(negative_demand) > 0:
        print(f"\n发现 {len(negative_demand)} 条负数需求量数据")
        print("\n负数需求量数据示例（前5条）：")
        print(negative_demand[['过账日期', '工厂编码', '物料编码', '需求量']].head())
        
        # 移除负数需求量的行
        demand_data = demand_data[demand_data['需求量'] >= 0]
        print(f"\n移除负数需求量数据后，剩余 {len(demand_data)} 条数据")
    
    return demand_data

def translate_feature_names(demand_data):
    """
    将特征名称翻译为中文
    
    Args:
        demand_data: 原始数据DataFrame
        
    Returns:
        特征名称翻译后的DataFrame
    """
    # 定义特征名称映射
    feature_mapping = {
        'month': '月份',
        'day': '日期',
        'dayofweek': '星期几',
        'quarter': '季度',
        'is_month_start': '是否月初',
        'is_month_end': '是否月末',
        '工厂编码': '工厂编码',
        '物料编码': '物料编码',
        '需求量': '需求量'
    }
    
    # 添加滞后特征的中文名称
    for lag in [1, 3, 7, 14, 30]:
        feature_mapping[f'lag_{lag}d'] = f'{lag}天前需求量'
        feature_mapping[f'lag_{lag}d_change'] = f'{lag}天前需求量变化率'
    
    # 添加滑动窗口特征的中文名称
    for window in [3, 7, 14, 30]:
        feature_mapping[f'rolling_mean_{window}d'] = f'{window}天平均需求量'
        feature_mapping[f'rolling_std_{window}d'] = f'{window}天需求量标准差'
        feature_mapping[f'rolling_min_{window}d'] = f'{window}天最小需求量'
        feature_mapping[f'rolling_max_{window}d'] = f'{window}天最大需求量'
        feature_mapping[f'rolling_mean_{window}d_change'] = f'{window}天平均需求量变化率'
    
    # 重命名列
    demand_data = demand_data.rename(columns=feature_mapping)
    
    return demand_data

def add_time_features(demand_data):
    """
    添加时间相关特征
    
    Args:
        demand_data: 原始数据DataFrame
        
    Returns:
        添加了时间特征的DataFrame
    """
    # 创建DataFrame的副本以避免SettingWithCopyWarning
    df = demand_data.copy()
    
    # 基本时间特征
    df.loc[:, '月份'] = df['过账日期'].dt.month
    df.loc[:, '日期'] = df['过账日期'].dt.day
    df.loc[:, '星期几'] = df['过账日期'].dt.dayofweek
    df.loc[:, '季度'] = df['过账日期'].dt.quarter
    df.loc[:, '是否月初'] = df['过账日期'].dt.is_month_start.astype(int)
    df.loc[:, '是否月末'] = df['过账日期'].dt.is_month_end.astype(int)
    
    return df

def safe_pct_change(x):
    """
    安全计算百分比变化，处理除零和无穷大的情况
    
    Args:
        x: 输入序列
        
    Returns:
        处理后的百分比变化
    """
    # 使用fill_method=None来避免FutureWarning
    pct = x.pct_change(fill_method=None)
    # 将无穷大替换为0
    pct = pct.replace([np.inf, -np.inf], 0)
    # 将NaN替换为0
    pct = pct.fillna(0)
    return pct

def add_lag_features(demand_data, lag_days=[1, 3, 7, 14, 30]):
    """
    添加滞后特征
    
    Args:
        demand_data: 原始数据DataFrame
        lag_days: 滞后天数列表
        
    Returns:
        添加了滞后特征的DataFrame
    """
    # 创建DataFrame的副本以避免SettingWithCopyWarning
    df = demand_data.copy()
    
    # 按工厂编码、物料编码和日期排序
    df = df.sort_values(['工厂编码', '物料编码', '过账日期'])
    
    # 为每个滞后天数创建特征
    for lag in lag_days:
        # 创建滞后特征
        df.loc[:, f'{lag}天前需求量'] = df.groupby(['工厂编码', '物料编码'])['需求量'].shift(lag)
        
        # 创建滞后变化率特征
        df.loc[:, f'{lag}天前需求量变化率'] = df.groupby(['工厂编码', '物料编码'])[f'{lag}天前需求量'].transform(safe_pct_change)
    
    return df

def add_rolling_features(demand_data, windows=[3, 7, 14, 30]):
    """
    添加滑动窗口特征
    
    Args:
        demand_data: 原始数据DataFrame
        windows: 窗口大小列表
        
    Returns:
        添加了滑动窗口特征的DataFrame
    """
    # 创建DataFrame的副本以避免SettingWithCopyWarning
    df = demand_data.copy()
    
    # 按工厂编码、物料编码和日期排序
    df = df.sort_values(['工厂编码', '物料编码', '过账日期'])
    
    # 为每个窗口大小创建特征
    for window in windows:
        # 计算滑动窗口统计量
        df.loc[:, f'{window}天平均需求量'] = df.groupby(['工厂编码', '物料编码'])['需求量'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
        df.loc[:, f'{window}天需求量标准差'] = df.groupby(['工厂编码', '物料编码'])['需求量'].transform(
            lambda x: x.rolling(window=window, min_periods=1).std()
        )
        df.loc[:, f'{window}天最小需求量'] = df.groupby(['工厂编码', '物料编码'])['需求量'].transform(
            lambda x: x.rolling(window=window, min_periods=1).min()
        )
        df.loc[:, f'{window}天最大需求量'] = df.groupby(['工厂编码', '物料编码'])['需求量'].transform(
            lambda x: x.rolling(window=window, min_periods=1).max()
        )
        
        # 计算滑动窗口变化率
        df.loc[:, f'{window}天平均需求量变化率'] = df.groupby(['工厂编码', '物料编码'])[f'{window}天平均需求量'].transform(safe_pct_change)
    
    return df

def normalize_features(demand_data):
    """
    对数值特征进行归一化处理
    
    Args:
        demand_data: 原始数据DataFrame
        
    Returns:
        归一化后的DataFrame
    """
    # 创建DataFrame的副本以避免SettingWithCopyWarning
    df = demand_data.copy()
    
    # 获取所有数值列
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    # 对每个数值列进行归一化
    for col in numeric_columns:
        if col not in ['过账日期', '需求量', '工厂编码', '物料编码', '物料品牌', '物料类型', '物料品类', '月份', '日期', '星期几', '季度', '是否月初', '是否月末']:  # 排除分类特征
            # 计算均值和标准差
            mean_val = df[col].mean()
            std_val = df[col].std()
            
            # 避免除零
            if std_val != 0:
                df.loc[:, col] = (df[col] - mean_val) / std_val
            else:
                df.loc[:, col] = df[col] - mean_val
            
            # 处理无穷大和NaN
            df.loc[:, col] = df[col].replace([np.inf, -np.inf], 0)
            df.loc[:, col] = df[col].fillna(0)
    
    return df

def preprocess_data(demand_data):
    """
    数据预处理函数
    
    Args:
        demand_data: 原始数据DataFrame
        
    Returns:
        处理后的DataFrame
    """
    print("\n" + "="*50)
    print("开始数据预处理流程")
    print("="*50)
    
    # 1. 日期转换
    print("\n[1/7] 日期转换")
    print("开始转换日期格式...")
    demand_data['过账日期'] = pd.to_datetime(demand_data['过账日期'], errors='coerce')
    # 检查是否有无效日期
    if demand_data['过账日期'].isna().any():
        raise ValueError("过账日期列包含无效日期，请检查数据格式")
    print("日期转换完成")
    
    # 2. 处理负数需求量
    print("\n[2/7] 处理负数需求量")
    print("开始处理负数需求量...")
    demand_data = handle_negative_demand(demand_data)
    print("负数需求量处理完成")
    
    # 3. 去除重复数据
    print("\n[3/7] 去除重复数据")
    print("开始去除重复数据...")
    demand_data = remove_duplicates(demand_data)
    print("重复数据去除完成")
    
    # 4. 添加时间特征
    print("\n[4/7] 添加时间特征")
    print("开始添加时间特征...")
    demand_data = add_time_features(demand_data)
    print("时间特征添加完成")
    
    # 5. 添加滞后特征
    print("\n[5/7] 添加滞后特征")
    print("开始添加滞后特征...")
    demand_data = add_lag_features(demand_data)
    print("滞后特征添加完成")
    
    # 6. 添加滑动窗口特征
    print("\n[6/7] 添加滑动窗口特征")
    print("开始添加滑动窗口特征...")
    demand_data = add_rolling_features(demand_data)
    print("滑动窗口特征添加完成")
    
    # 7. 归一化特征
    print("\n[7/7] 归一化特征")
    print("开始归一化特征...")
    demand_data = normalize_features(demand_data)
    print("特征归一化完成")
    
    # 删除包含NaN的行
    print("\n清理数据...")
    demand_data = demand_data.dropna()
    print(f"清理完成，最终数据量: {len(demand_data)} 条")
    
    print("\n特征列表：")
    print(demand_data.columns.tolist())
    print("\n数据预览：")
    print(demand_data.head())
    
    print("\n" + "="*50)
    print("数据预处理完成")
    print("="*50 + "\n")
    
    return demand_data

def split_data(demand_data):
    """
    划分后三个月的数据为测试集，其余为训练集
    """
    print("\n" + "="*50)
    print("开始数据集划分")
    print("="*50)
    
    # 取最后3个月作为测试集
    cutoff_date = demand_data["过账日期"].max() - pd.DateOffset(months=3)
    train_set = demand_data[demand_data["过账日期"] <= cutoff_date]
    test_set = demand_data[demand_data["过账日期"] > cutoff_date]

    # 获取所有特征列（除了目标变量和日期列）
    feature_columns = [col for col in demand_data.columns 
                      if col not in ['过账日期', '需求量']]

    # 分离训练集特征和目标
    X_train = train_set[feature_columns]
    y_train = train_set['需求量']

    # 分离测试集特征和目标
    X_test = test_set[feature_columns]
    y_test = test_set['需求量']
    
    print(f"\n训练集大小: {len(X_train)} 样本")
    print(f"测试集大小: {len(X_test)} 样本")
    print(f"特征数量: {len(feature_columns)}")
    print("\n特征列表:")
    print(feature_columns)
    
    print("\n" + "="*50)
    print("数据集划分完成")
    print("="*50 + "\n")
    
    return X_train, y_train, X_test, y_test

def save_preprocessed_data(data, output_path):
    """
    Save the preprocessed data to a CSV file.
    
    Parameters:
    data (DataFrame): The preprocessed data.
    output_path (str): The path to save the preprocessed data CSV file.
    """
    data.to_csv(output_path, index=False)

