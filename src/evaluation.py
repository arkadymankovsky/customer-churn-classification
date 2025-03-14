"""
Model evaluation metrics and functions.
"""
import json
from typing import Dict, Any
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from . import config

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None) -> Dict[str, float]:
    """
    Calculate various classification metrics.
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        y_prob (np.ndarray, optional): Predicted probabilities
        
    Returns:
        Dict[str, float]: Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred)
    }
    
    if y_prob is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_prob[:, 1])
    
    conf_matrix = confusion_matrix(y_true, y_pred)
    metrics['true_negatives'] = conf_matrix[0, 0]
    metrics['false_positives'] = conf_matrix[0, 1]
    metrics['false_negatives'] = conf_matrix[1, 0]
    metrics['true_positives'] = conf_matrix[1, 1]
    
    return metrics

def save_metrics(metrics: Dict[str, Any], path: str = config.RESULTS_PATH) -> None:
    """
    Save metrics to a JSON file.
    
    Args:
        metrics (Dict[str, Any]): Metrics to save
        path (str): Path to save the metrics
    """
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=4) 