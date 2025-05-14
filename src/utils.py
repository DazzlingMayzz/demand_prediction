def load_csv(file_path):
    import pandas as pd
    return pd.read_csv(file_path)

def save_csv(dataframe, file_path):
    dataframe.to_csv(file_path, index=False)

def format_predictions(predictions):
    """
    将预测结果格式化为保留两位小数
    
    参数:
    predictions (DataFrame): 包含预测结果的DataFrame
    
    返回:
    DataFrame: 格式化后的DataFrame，所有数值列保留两位小数
    """
    # 复制DataFrame以避免修改原始数据
    formatted = predictions.copy()
    
    # 找出所有数值列
    numeric_columns = formatted.select_dtypes(include=['float64', 'int64']).columns
    
    # 对所有数值列应用round函数，保留两位小数
    for col in numeric_columns:
        formatted[col] = formatted[col].round(2)
    
    return formatted

def visualize_demand(data):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.plot(data['posting_date'], data['demand_quantity'], marker='o')
    plt.title('Historical Demand Over Time')
    plt.xlabel('Date')
    plt.ylabel('Demand Quantity')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def aggregate_daily_to_monthly(daily_predictions):
    """
    将每日需求量预测结果聚合成每月需求量
    
    参数:
    daily_predictions (DataFrame): 包含每日预测需求量的DataFrame，必须包含以下列：
        - 工厂编码
        - 物料编码
        - 预测日期 (datetime格式)
        - 预测需求量
    
    返回:
    DataFrame: 包含每月预测需求量的DataFrame，包含以下列：
        - 工厂编码
        - 物料编码
        - 预测月份 (YYYY-MM格式)
        - 预测需求量 (每月总量)
    """
    import pandas as pd
    
    # 确保预测日期是datetime格式
    if not pd.api.types.is_datetime64_any_dtype(daily_predictions['预测日期']):
        daily_predictions['预测日期'] = pd.to_datetime(daily_predictions['预测日期'])
    
    # 提取年月
    daily_predictions['预测月份'] = daily_predictions['预测日期'].dt.strftime('%Y-%m')
    
    # 按工厂编码、物料编码和预测月份分组，计算每月需求量总和
    monthly_predictions = daily_predictions.groupby(
        ['工厂编码', '物料编码', '预测月份']
    )['预测需求量'].sum().reset_index()
    
    # 格式化预测需求量，保留两位小数
    monthly_predictions['预测需求量'] = monthly_predictions['预测需求量'].round(2)
    
    return monthly_predictions

def reshape_monthly_predictions(monthly_predictions):
    """
    将每月预测结果重塑为宽格式，每个月份一列
    
    参数:
    monthly_predictions (DataFrame): 包含每月预测需求量的DataFrame，必须包含以下列：
        - 工厂编码
        - 物料编码
        - 预测月份
        - 预测需求量
    
    返回:
    DataFrame: 重塑后的DataFrame，包含以下列：
        - 工厂编码
        - 物料编码
        - M+1月预测需求量
        - M+2月预测需求量
        - M+3月预测需求量
    """
    
    # 将预测结果重塑为宽格式，每个月份一列
    predictions_wide = monthly_predictions.pivot_table(
        index=['工厂编码', '物料编码'],
        columns='预测月份',
        values='预测需求量',
        aggfunc='first'
    ).reset_index()
    
    # 重命名列
    if '2020-12' in predictions_wide.columns:
        predictions_wide.rename(columns={'2020-12': 'M+1月预测需求量'}, inplace=True)
    if '2021-01' in predictions_wide.columns:
        predictions_wide.rename(columns={'2021-01': 'M+2月预测需求量'}, inplace=True)
    if '2021-02' in predictions_wide.columns:
        predictions_wide.rename(columns={'2021-02': 'M+3月预测需求量'}, inplace=True)
    
    # 确保所有预测需求量列保留两位小数
    for col in ['M+1月预测需求量', 'M+2月预测需求量', 'M+3月预测需求量']:
        if col in predictions_wide.columns:
            predictions_wide[col] = predictions_wide[col].round(2)
    
    return predictions_wide