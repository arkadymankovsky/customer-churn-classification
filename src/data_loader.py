"""
Functions for loading and preprocessing data.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from . import config

def load_data(file_path: str = config.RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load the raw data from CSV file.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded data
    """
    return pd.read_csv(file_path)

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the raw data.
    
    Args:
        df (pd.DataFrame): Raw data
        
    Returns:
        pd.DataFrame: Preprocessed data
    """
    # Add preprocessing steps here
    return df

def split_data(df: pd.DataFrame, target_col: str = config.TARGET_COLUMN):
    """
    Split data into training and testing sets.
    
    Args:
        df (pd.DataFrame): Input data
        target_col (str): Name of target column
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    return train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE
    )