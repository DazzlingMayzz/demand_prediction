import pandas as pd

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

def preprocess_data(demand_data):
    """
    数据预处理函数
    
    Args:
        demand_data: 原始数据DataFrame
        
    Returns:
        处理后的DataFrame
    """
    # 将 '过账日期' 转换为 datetime 类型，处理无效日期
    demand_data['过账日期'] = pd.to_datetime(demand_data['过账日期'], errors='coerce')
    # 检查是否有无效日期
    if demand_data['过账日期'].isna().any():
        raise ValueError("过账日期列包含无效日期，请检查数据格式")
    # 提取月份和日期
    demand_data['month'] = demand_data['过账日期'].dt.month
    demand_data['day'] = demand_data['过账日期'].dt.day
    print("Columns after preprocessing:", demand_data.columns)  # 打印列名
    print("Data types after preprocessing:", demand_data.iloc[0:2])  # 打印数据类型
    return demand_data

def split_data(demand_data):
    """
    划分后三个月的数据为测试集，其余为训练集
    """
    # 取最后3个月作为测试集
    cutoff_date = demand_data["过账日期"].max() - pd.DateOffset(months=3)
    train_set = demand_data[demand_data["过账日期"] <= cutoff_date]
    test_set = demand_data[demand_data["过账日期"] > cutoff_date]

    # 分离训练集特征和目标
    X_train = train_set[['month', 'day', '工厂编码', '物料编码']]
    y_train = train_set['需求量']

    # 分离测试集特征和目标
    X_test = test_set[['month', 'day', '工厂编码', '物料编码']]
    y_test = test_set['需求量']
    
    print(f"训练集大小: {len(X_train)} 样本")
    print(f"测试集大小: {len(X_test)} 样本")
    
    return X_train, y_train, X_test, y_test


    

def save_preprocessed_data(data, output_path):
    """
    Save the preprocessed data to a CSV file.
    
    Parameters:
    data (DataFrame): The preprocessed data.
    output_path (str): The path to save the preprocessed data CSV file.
    """
    data.to_csv(output_path, index=False)