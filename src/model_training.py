import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import numpy as np
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
import lightgbm as lgb
from model_evaluation import evaluate_model
import os
from datetime import datetime
import time

def train_model(X_train, y_train, X_test, y_test):
    """
    训练模型函数
    
    Args:
        X_train: 训练数据特征
        y_train: 训练数据标签
        X_test: 测试数据特征
        y_test: 测试数据标签
        
    Returns:
        训练好的模型
    """
    print("开始模型训练...")
    print(f"训练集大小: {X_train.shape}")
    print(f"测试集大小: {X_test.shape}")
    print(f"数据特征：{X_train.columns}")
    
    # 创建输出目录
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 创建评估结果列表
    evaluation_results = []
    
    # 1. 定义多个模型
    models = {
        'random_forest': RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            verbose=0  # 关闭详细输出
        ),
        'xgboost': XGBRegressor(
            n_estimators=200,
            max_depth=7,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbosity=0  # 关闭详细输出
        ),
        'lightgbm': LGBMRegressor(
            n_estimators=100, 
            max_depth=5,
            learning_rate=0.2,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,         # 使用所有CPU核心
            verbose=-1,        # 关闭详细输出
            num_leaves=31,     # 控制叶子节点数量
            min_data_in_leaf=20,  # 控制叶子节点最小数据量
            max_bin=255,       # 控制特征分箱数量
            device='cpu'       # 明确使用CPU
        )
    }
    
    # 2. 训练并评估每个模型
    best_model = None
    best_score = float('-inf')
    
    for name, model in models.items():
        print(f"\n训练 {name} 模型...")
        
        # 记录训练开始时间
        train_start_time = time.time()
        
        # 对于LightGBM，使用早停策略
        if name == 'lightgbm':
            eval_set = [(X_test, y_test)]
            model.fit(
                X_train, y_train,
                eval_set=eval_set,
                eval_metric='rmse',
                callbacks=[lgb.early_stopping(10)],  # 使用回调函数实现早停
            )
        else:
            model.fit(X_train, y_train)
        
        # 记录训练结束时间
        train_end_time = time.time()
        train_time = train_end_time - train_start_time
        
        # 计算训练集和测试集的得分
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        # 计算均方误差
        train_rmse, train_mae = evaluate_model(y_train, model.predict(X_train))
        test_rmse, test_mae = evaluate_model(y_test, model.predict(X_test))
        
        print(f"{name} 模型:")
        print(f"训练集 R2 分数: {train_score:.4f}")
        print(f"测试集 R2 分数: {test_score:.4f}")
        print(f"训练时间: {train_time:.2f} 秒")
        
        # 收集评估结果
        evaluation_results.append({
            '模型名称': name,
            '训练集_R2': train_score,
            '测试集_R2': test_score,
            '训练集_RMSE': train_rmse,
            '测试集_RMSE': test_rmse,
            '训练集_MAE': train_mae,
            '测试集_MAE': test_mae,
            '训练时间(秒)': train_time
        })
        
        # 更新最佳模型
        if test_score > best_score:
            best_score = test_score
            best_model = model
            print(f"当前最佳模型: {name}")
    
    # 将评估结果保存到文件
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    evaluation_df = pd.DataFrame(evaluation_results)
    evaluation_file = os.path.join(output_dir, f'model_evaluation_{timestamp}.csv')
    evaluation_df.to_csv(evaluation_file, index=False, encoding='utf-8-sig')
    print(f"\n模型评估结果已保存到: {evaluation_file}")
    
    # 3. 特征重要性分析
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        feature_names = X_train.columns
        
        # 创建特征重要性DataFrame
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        })
        
        # 按重要性排序
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        
        # 打印前10个最重要的特征
        print("\n前10个最重要的特征:")
        print(feature_importance.head(10))
        
        # 保存特征重要性到文件
        feature_importance.to_csv('data/feature_importance.csv', index=False)
    
    print("\n模型训练完成")
    return best_model

def save_model(model, filename):
    print(f"正在保存模型到 {filename}...")
    joblib.dump(model, filename)
    print("模型保存完成")
