"""
Functions for preparing data for modeling (splitting, cross-validation, etc.).
"""
import pandas as pd
import numpy as np
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from . import config, model_config

def split_data(
    df: pd.DataFrame,
    target_col: str = model_config.TARGET_COLUMN,
    test_size: float = model_config.TEST_SIZE,
    stratify: bool = True,
    random_state: int = model_config.RANDOM_STATE
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
    X = df[model_config.NUMERICAL_FEATURES + model_config.CATEGORICAL_FEATURES]
    y = df[target_col]
    
    stratify_param = y if stratify else None
    
    return train_test_split(
        X, y,
        test_size=test_size,
        stratify=stratify_param,
        random_state=random_state
    )

def create_folds(
    X: pd.DataFrame,
    y: Optional[pd.Series] = None,
    n_splits: int = 5,
    stratify: bool = True,
    shuffle: bool = True,
    random_state: int = model_config.RANDOM_STATE
) -> pd.DataFrame:
    """
    Create cross-validation folds and add fold indices to the dataframe.
    Can handle both:
    1. A single DataFrame with target column (old behavior)
    2. Separate feature matrix X and target vector y (new behavior)
    
    Args:
        X (pd.DataFrame): Either complete DataFrame with target column or feature matrix
        y (Optional[pd.Series]): Target vector if X is feature matrix, None if X is complete DataFrame
        n_splits (int): Number of folds
        stratify (bool): Whether to maintain class distribution in folds
        shuffle (bool): Whether to shuffle before splitting
        random_state (int): Random seed for reproducibility
        
    Returns:
        pd.DataFrame: DataFrame with additional 'fold' column
    """
    # Create a copy to avoid modifying the original data
    X = X.copy()
    
    if stratify:
        kf = StratifiedKFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state
        )
        # If y is provided separately, use it directly
        target_for_split = y if y is not None else X[config.TARGET_COLUMN]
        splits = kf.split(X, target_for_split)
    else:
        kf = KFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state
        )
        splits = kf.split(X)
    
    # Add fold column to X
    X['fold'] = -1
    for fold, (_, val_idx) in enumerate(splits):
        X.loc[val_idx, 'fold'] = fold
    
    return X 