"""
Functions for loading data from various sources.
"""
import pandas as pd
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

def load_processed_data(file_path: str = config.PROCESSED_DATA_PATH) -> pd.DataFrame:
    """
    Load previously processed data from CSV file.
    
    Args:
        file_path (str): Path to the processed CSV file
        
    Returns:
        pd.DataFrame: Processed data
    """
    return pd.read_csv(file_path)