import pandas as pd
from data_preprocessing import load_data, preprocess_data, split_data
from model_training import train_model, save_model
from model_predict import predict_future_demand
import os
import time

def main():
    start_time = time.time()
    print("=" * 50)
    print("离散制造行业产品物料需求智能预测系统")
    print("=" * 50)
    
    # 创建models目录（如果不存在）
    if not os.path.exists('models'):
        print("创建models目录...")
        os.makedirs('models')
    
    # 加载历史需求数据
    print("\n[1/4] 数据加载阶段")
    demand_data = load_data('data/demand_train.csv')

    # 预处理数据
    print("\n[2/4] 数据预处理阶段")
    processed_demand = preprocess_data(demand_data)

    # 训练预测模型
    print("\n[3/4] 模型训练阶段")
    model = train_model(*split_data(processed_demand))
    
    # 保存模型
    model_path = 'models/forecast_model.joblib'
    save_model(model, model_path)
    print(f"模型已保存到 {model_path}")

    # 预测未来3个月的需求
    print("\n[4/4] 需求预测阶段")
    predictions = predict_future_demand(model, processed_demand)
    
    # 计算总耗时
    end_time = time.time()
    total_time = end_time - start_time
    minutes = int(total_time // 60)
    seconds = int(total_time % 60)
    
    print("\n" + "=" * 50)
    print(f"预测流程完成! 总耗时: {minutes}分{seconds}秒")
    print("=" * 50)

if __name__ == "__main__":
    main()