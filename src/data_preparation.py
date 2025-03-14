"""
Functions for preparing data for modeling (splitting, cross-validation, etc.).
"""
import pandas as pd
import numpy as np
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from . import config

def split_data(
    df: pd.DataFrame,
    target_col: str = config.TARGET_COLUMN,
    test_size: float = config.TEST_SIZE,
    stratify: bool = True,
    random_state: int = config.RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into training and testing sets.
    
    Args:
        df (pd.DataFrame): Input data
        target_col (str): Name of target column
        test_size (float): Proportion of data to use for testing
        stratify (bool): Whether to maintain class distribution in split
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    stratify_param = y if stratify else None
    
    return train_test_split(
        X, y,
        test_size=test_size,
        stratify=stratify_param,
        random_state=random_state
    )

def create_folds(
    df: pd.DataFrame,
    target_col: str = config.TARGET_COLUMN,
    n_splits: int = 5,
    stratify: bool = True,
    shuffle: bool = True,
    random_state: int = config.RANDOM_STATE
) -> pd.DataFrame:
    """
    Create cross-validation folds and add fold indices to the dataframe.
    
    Args:
        df (pd.DataFrame): Input data
        target_col (str): Name of target column
        n_splits (int): Number of folds
        stratify (bool): Whether to maintain class distribution in folds
        shuffle (bool): Whether to shuffle before splitting
        random_state (int): Random seed for reproducibility
        
    Returns:
        pd.DataFrame: Original dataframe with additional 'fold' column
    """
    df = df.copy()
    
    if stratify:
        kf = StratifiedKFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state
        )
        splits = kf.split(df, df[target_col])
    else:
        kf = KFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state
        )
        splits = kf.split(df)
    
    df['fold'] = -1
    for fold, (_, val_idx) in enumerate(splits):
        df.loc[val_idx, 'fold'] = fold
    
    return df 