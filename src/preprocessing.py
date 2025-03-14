"""
Data preprocessing and cleaning functions.
"""
import pandas as pd
import numpy as np
from typing import Dict, List
from . import config

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the raw data by cleaning, handling missing values,
    and preparing it for feature engineering.
    
    Args:
        df (pd.DataFrame): Raw data
        
    Returns:
        pd.DataFrame: Preprocessed data
    """
    df = df.copy()
    
    # Add preprocessing steps here, for example:
    # - Handle missing values
    # - Remove duplicates
    # - Fix data types
    # - Handle outliers
    # - Basic feature transformations
    
    return df

def handle_missing_values(df: pd.DataFrame, strategy: Dict[str, str] = None) -> pd.DataFrame:
    """
    Handle missing values in the dataset.
    
    Args:
        df (pd.DataFrame): Input data
        strategy (Dict[str, str], optional): Dictionary mapping column names to imputation strategies
            ('mean', 'median', 'mode', 'constant', etc.)
            
    Returns:
        pd.DataFrame: Data with handled missing values
    """
    df = df.copy()
    if strategy is None:
        strategy = {}
    
    for column in df.columns:
        if df[column].isnull().any():
            if column in strategy:
                if strategy[column] == 'mean':
                    df[column].fillna(df[column].mean(), inplace=True)
                elif strategy[column] == 'median':
                    df[column].fillna(df[column].median(), inplace=True)
                elif strategy[column] == 'mode':
                    df[column].fillna(df[column].mode()[0], inplace=True)
                elif isinstance(strategy[column], (int, float, str)):
                    df[column].fillna(strategy[column], inplace=True)
            else:
                # Default strategy: use median for numerical, mode for categorical
                if np.issubdtype(df[column].dtype, np.number):
                    df[column].fillna(df[column].median(), inplace=True)
                else:
                    df[column].fillna(df[column].mode()[0], inplace=True)
    
    return df

def remove_duplicates(df: pd.DataFrame, subset: List[str] = None) -> pd.DataFrame:
    """
    Remove duplicate rows from the dataset.
    
    Args:
        df (pd.DataFrame): Input data
        subset (List[str], optional): List of columns to consider for duplicates
            
    Returns:
        pd.DataFrame: Data with duplicates removed
    """
    return df.drop_duplicates(subset=subset)

def handle_outliers(df: pd.DataFrame, columns: List[str], method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
    """
    Handle outliers in specified numerical columns.
    
    Args:
        df (pd.DataFrame): Input data
        columns (List[str]): List of columns to check for outliers
        method (str): Method to use ('iqr' or 'zscore')
        threshold (float): Threshold for outlier detection
            
    Returns:
        pd.DataFrame: Data with handled outliers
    """
    df = df.copy()
    
    for column in columns:
        if method == 'iqr':
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            # Cap outliers at bounds
            df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
            
        elif method == 'zscore':
            z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
            df.loc[z_scores > threshold, column] = df[column].mean()
    
    return df 