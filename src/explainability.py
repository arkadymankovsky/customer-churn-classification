"""
Model explanation functions using SHAP values.
"""
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Tuple

def calculate_shap_values(model: Any, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate SHAP values for model predictions.
    
    Args:
        model: Trained model instance
        X (pd.DataFrame): Input features
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: Base value and SHAP values
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # For binary classification, take positive class
    
    return explainer.expected_value, shap_values

def plot_feature_importance(shap_values: np.ndarray, feature_names: list, save_path: str = None):
    """
    Create and save SHAP feature importance plot.
    
    Args:
        shap_values (np.ndarray): SHAP values
        feature_names (list): List of feature names
        save_path (str, optional): Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, feature_names=feature_names, show=False)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def get_feature_importance(shap_values: np.ndarray, feature_names: list) -> pd.DataFrame:
    """
    Get global feature importance based on SHAP values.
    
    Args:
        shap_values (np.ndarray): SHAP values
        feature_names (list): List of feature names
        
    Returns:
        pd.DataFrame: Feature importance scores
    """
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': np.abs(shap_values).mean(axis=0)
    })
    
    return importance_df.sort_values('importance', ascending=False) 