import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import numpy as np
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
import lightgbm as lgb
from src.model_evaluation import evaluate_model, visualize_predictions, save_prediction_with_features
import os
from datetime import datetime
import time
from tabulate import tabulate

def normalize_feature_importance(importances):
    """
    归一化特征重要性
    
    Args:
        importances: 原始特征重要性数组
        
    Returns:
        归一化后的特征重要性数组
    """
    # 确保所有值都是非负的
    importances = np.abs(importances)
    # 归一化到0-1范围
    if importances.sum() > 0:
        importances = importances / importances.sum()
    return importances

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
    models_dir = 'models'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    # 创建评估结果列表
    evaluation_results = []
    
    # 获取当前时间戳
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
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
    best_model_name = None
    trained_models = {}
    
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
        
        # 确保预测值非负
        train_predictions = np.maximum(model.predict(X_train), 0)
        test_predictions = np.maximum(model.predict(X_test), 0)
        
        # 计算均方误差
        train_rmse, train_mae = evaluate_model(y_train, train_predictions, step='train')
        test_rmse, test_mae = evaluate_model(y_test, test_predictions, step='test')
        
        # 可视化预测结果
        print(f"\n{name} 模型预测结果可视化：")
        # 保存验证集预测结果图表
        viz_path = os.path.join(output_dir, f'{name}_predictions_{timestamp}.png')
        visualize_predictions(
            y_test, 
            test_predictions,
            model_name=name,
            save_path=viz_path
        )
        
        # 保存预测结果（含原始特征）
        prediction_path = os.path.join(output_dir, f'{name}_test_predictions_{timestamp}.csv')
        save_prediction_with_features(
            X_test,
            y_test,
            test_predictions.round(2),
            model_name=name,
            save_path=prediction_path
        )
        
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
        
        # 保存模型
        model_filename = f'{name}_{timestamp}.joblib'
        model_path = os.path.join(models_dir, model_filename)
        joblib.dump(model, model_path)
        print(f"模型已保存到: {model_path}")
        
        # 显示特征重要性（如果模型支持）
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_names = X_train.columns
            
            # 归一化特征重要性
            importances = normalize_feature_importance(importances)
            
            # 创建特征重要性DataFrame
            feature_importance = pd.DataFrame({
                '特征': feature_names,
                '重要性': importances
            })
            
            # 按重要性排序
            feature_importance = feature_importance.sort_values('重要性', ascending=False)
            
            # 显示特征重要性表格
            print(f"\n{name} 模型特征重要性:")
            print(tabulate(
                feature_importance,
                headers='keys',
                tablefmt='grid',
                floatfmt='.4f',
                showindex=False
            ))
            
            # 保存特征重要性到文件
            importance_path = os.path.join(output_dir, f'{name}_feature_importance_{timestamp}.csv')
            feature_importance.to_csv(importance_path, index=False, encoding='utf-8-sig')
            print(f"特征重要性已保存到: {importance_path}")
        
        # 更新最佳模型
        if test_score > best_score:
            best_score = test_score
            best_model = model
            best_model_name = name
            print(f"当前最佳模型: {name}")
    
    # 将评估结果保存到文件
    evaluation_df = pd.DataFrame(evaluation_results)
    evaluation_file = os.path.join(output_dir, f'model_evaluation_{timestamp}.csv')
    evaluation_df.to_csv(evaluation_file, index=False, encoding='utf-8-sig')
    
    # 打印评估结果表格
    print("\n" + "="*80)
    print("模型评估结果汇总")
    print("="*80)
    
    # 格式化评估结果表格
    summary_table = evaluation_df.copy()
    # 重命名列以使其更易读
    summary_table = summary_table.rename(columns={
        '模型名称': '模型',
        '训练集_R2': '训练集 R²',
        '测试集_R2': '测试集 R²',
        '训练集_RMSE': '训练集 RMSE',
        '测试集_RMSE': '测试集 RMSE',
        '训练集_MAE': '训练集 MAE',
        '测试集_MAE': '测试集 MAE',
        '训练时间(秒)': '训练时间(秒)'
    })
    
    # 设置显示格式
    pd.set_option('display.float_format', lambda x: '%.4f' % x)
    
    # 打印表格
    print("\n", tabulate(
        summary_table,
        headers='keys',
        tablefmt='grid',
        showindex=False,
        floatfmt='.4f'
    ))
    
    print("\n" + "="*80)
    print(f"最佳模型: {best_model_name}")
    print(f"最佳测试集 R² 分数: {best_score:.4f}")
    print("="*80 + "\n")
    
    print("\n模型训练完成")
    return best_model, trained_models
