import pandas as pd
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
import numpy as np

def evaluate_model(y_true, y_pred):
    """
    Evaluate the performance of the forecasting model using RMSE and MAE.

    Parameters:
    y_true (array-like): True demand values.
    y_pred (array-like): Predicted demand values.

    Returns:
    dict: A dictionary containing RMSE and MAE values.
    """
    rmse = np.sqrt(root_mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    print(f"模型效果评估：\nRMSE:{rmse}\nMAE:{mae}")
    
    return rmse,mae

def print_evaluation_results(evaluation_results):
    """
    Print the evaluation results in a readable format.

    Parameters:
    evaluation_results (dict): A dictionary containing evaluation metrics.
    """
    print("Model Evaluation Results:")
    for metric, value in evaluation_results.items():
        print(f"{metric}: {value:.4f}")