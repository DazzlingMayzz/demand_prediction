import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import os
from utils import aggregate_daily_to_monthly, reshape_monthly_predictions, format_predictions
from tqdm import tqdm

def predict_future_demand(model, demand_data, output_path='output/predictions.csv'):
    """
    预测未来3个月（2020年12月、2021年1月、2021年2月）每个工厂对应的每个产品物料的每日需求量，
    并将结果聚合成每月需求量
    
    参数:
    model: 训练好的模型
    demand_data: 历史需求数据
    output_path: 输出预测结果的路径
    
    返回:
    DataFrame: 包含预测结果的DataFrame
    """
    print("开始预测未来需求...")
    
    # 获取所有唯一的工厂编码和物料编码组合
    unique_combinations = demand_data[['工厂编码', '物料编码']].drop_duplicates()
    total_combinations = len(unique_combinations)
    print(f"共有 {total_combinations} 个工厂-物料组合需要预测")
    
    # 创建未来3个月的日期
    future_months = [
        datetime(2020, 12, 1),
        datetime(2021, 1, 1),
        datetime(2021, 2, 1)
    ]
    
    # 创建预测结果DataFrame
    predictions = []
    
    # 使用tqdm显示预测进度
    for _, row in tqdm(unique_combinations.iterrows(), total=total_combinations, desc="预测进度"):
        factory_code = row['工厂编码']
        material_code = row['物料编码']
        
        # 对每个未来月份进行预测
        for month_date in future_months:
            # 获取该月的天数
            if month_date.month == 12:
                days_in_month = 31
            elif month_date.month == 1:
                days_in_month = 31
            elif month_date.month == 2:
                days_in_month = 28  # 2021年2月有28天
            else:
                days_in_month = 30
            
            # 对每一天进行预测
            for day in range(1, days_in_month + 1):
                # 创建特征
                features = pd.DataFrame({
                    'month': [month_date.month],
                    'day': [day],
                    '工厂编码': [factory_code],
                    '物料编码': [material_code]
                })
                
                # 预测需求量
                predicted_demand = model.predict(features)[0]
                
                # 创建日期
                prediction_date = datetime(month_date.year, month_date.month, day)
                
                # 添加到预测结果
                predictions.append({
                    '工厂编码': factory_code,
                    '物料编码': material_code,
                    '预测日期': prediction_date,
                    '预测需求量': round(predicted_demand, 2)  # 保留两位小数
                })
    
    print("预测完成，正在处理结果...")
    
    # 转换为DataFrame
    predictions_df = pd.DataFrame(predictions)
    
    # 保存每日预测结果
    print(f"正在保存每日预测结果到 {output_path+'tmp'}...")
    predictions_df.to_csv(output_path+'tmp', index=False)
    print("每日预测结果保存完成")
    
    # 将每日预测结果聚合成每月预测结果
    print("正在将每日预测结果聚合成每月预测结果...")
    monthly_predictions = aggregate_daily_to_monthly(predictions_df)
    
    # 将每月预测结果重塑为宽格式
    print("正在重塑预测结果为宽格式...")
    predictions_wide = reshape_monthly_predictions(monthly_predictions)
    
    # 格式化预测结果，确保所有数值列保留两位小数
    predictions_wide = format_predictions(predictions_wide)
    
    # 保存预测结果
    print(f"正在保存每月预测结果到 {output_path}...")
    predictions_wide.to_csv(output_path, index=False)
    print(f"预测结果已保存到 {output_path}")
    
    return predictions_wide

def load_model(model_path):
    """
    加载保存的模型
    
    参数:
    model_path: 模型文件路径
    
    返回:
    加载的模型
    """
    print(f"正在加载模型 {model_path}...")
    model = joblib.load(model_path)
    print("模型加载完成")
    return model

def main():
    """
    主函数，用于执行预测流程
    """
    # 加载模型
    model_path = 'models/forecast_model.joblib'
    if not os.path.exists(model_path):
        print(f"错误: 模型文件 {model_path} 不存在")
        return
    
    model = load_model(model_path)
    
    # 加载历史数据
    print("正在加载历史数据...")
    from data_preprocessing import load_data
    demand_data = load_data('data/demand_train.csv')
    print("历史数据加载完成")
    
    # 预测未来需求
    predictions = predict_future_demand(model, demand_data)
    
    print("预测流程完成!")

if __name__ == "__main__":
    main()
