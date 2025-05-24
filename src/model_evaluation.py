import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Microsoft YaHei']  # 优先使用这些字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def evaluate_model(y_true, y_pred, step):
    """
    使用RMSE和MAE评估预测模型的性能

    Parameters:
    y_true (array-like): 真实值
    y_pred (array-like): 预测值
    step (str): 'train' or 'test' 表示评估步骤

    Returns:
    tuple: (RMSE, MAE) 值
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    print(f"模型效果评估：step={step}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"真实值范围: [{y_true.min():.4f}, {y_true.max():.4f}]")
    print(f"预测值范围: [{y_pred.min():.4f}, {y_pred.max():.4f}]")
    
    return rmse, mae

def save_prediction_with_features(X, y_true, y_pred, model_name, save_path):
    """
    保存包含原始特征、实际值和预测值的对比表格
    
    Parameters:
    X (DataFrame): 原始特征数据
    y_true (array-like): 实际值
    y_pred (array-like): 预测值
    model_name (str): 模型名称
    save_path (str): 保存路径
    """
    # 创建结果DataFrame
    result_df = X.copy()
    result_df['实际值'] = y_true
    result_df['预测值'] = y_pred
    
    # 保存结果
    result_df.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"\n{model_name} 模型预测结果（含原始特征）已保存到: {save_path}")
    
    # 打印数据预览
    print("\n数据预览（前5行）:")
    print(result_df.head())

def visualize_predictions(y_true, y_pred, model_name, save_path=None):
    """
    可视化预测结果，包括散点图和残差图
    
    Parameters:
    y_true (array-like): 实际值
    y_pred (array-like): 预测值
    model_name (str): 模型名称
    save_path (str, optional): 保存图表的路径
    """
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. 散点图：实际值 vs 预测值
    ax1.scatter(y_true, y_pred, alpha=0.5)
    ax1.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    ax1.set_xlabel('实际值', fontsize=12)
    ax1.set_ylabel('预测值', fontsize=12)
    ax1.set_title(f'{model_name} - 预测值 - 实际值', fontsize=14)
    
    # 2. 残差图
    residuals = y_pred - y_true
    ax2.scatter(y_pred, residuals, alpha=0.5)
    ax2.axhline(y=0, color='r', linestyle='--')
    ax2.set_xlabel('预测值', fontsize=12)
    ax2.set_ylabel('残差 (预测值 - 实际值)', fontsize=12)
    ax2.set_title(f'{model_name} - 残差分布', fontsize=14)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存到: {save_path}")
    
    plt.show()

def print_evaluation_results(evaluation_results):
    """
    Print the evaluation results in a readable format.

    Parameters:
    evaluation_results (dict): A dictionary containing evaluation metrics.
    """
    print("Model Evaluation Results:")
    for metric, value in evaluation_results.items():
        print(f"{metric}: {value:.4f}")