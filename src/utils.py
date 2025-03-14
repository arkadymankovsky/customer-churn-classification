"""
Helper functions for the project.
"""
import os
import logging
from typing import Any
import pandas as pd
import numpy as np
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_directories(dirs: list) -> None:
    """
    Create directories if they don't exist.
    
    Args:
        dirs (list): List of directory paths
    """
    for dir_path in dirs:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            logger.info(f"Created directory: {dir_path}")

def save_predictions(df: pd.DataFrame, predictions: np.ndarray, 
                    probabilities: np.ndarray, output_path: str) -> None:
    """
    Save predictions and probabilities to CSV.
    
    Args:
        df (pd.DataFrame): Original data
        predictions (np.ndarray): Predicted labels
        probabilities (np.ndarray): Predicted probabilities
        output_path (str): Path to save results
    """
    results_df = df.copy()
    results_df['predicted_churn'] = predictions
    results_df['churn_probability'] = probabilities[:, 1]
    results_df['prediction_timestamp'] = datetime.now()
    
    results_df.to_csv(output_path, index=False)
    logger.info(f"Saved predictions to: {output_path}")

def load_or_create_experiment_tracking(path: str) -> pd.DataFrame:
    """
    Load or create experiment tracking DataFrame.
    
    Args:
        path (str): Path to experiment tracking file
        
    Returns:
        pd.DataFrame: Experiment tracking DataFrame
    """
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        df = pd.DataFrame(columns=[
            'experiment_id', 'timestamp', 'model_params',
            'accuracy', 'precision', 'recall', 'f1', 'roc_auc'
        ])
        df.to_csv(path, index=False)
        return df 