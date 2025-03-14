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

    df = remove_duplicates(df, subset=['customer_id', 'date'])
    df = add_binary_columns_for_missing_values(df)
    df = handle_missing_values(df)
    
    # Add preprocessing steps here, for example:
    # - Handle missing values
    # - Remove duplicates
    # - Fix data types
    # - Handle outliers
    # - Basic feature transformations
    
    return df

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in the dataset.
    
    Args:
        df (pd.DataFrame): Input data

    Returns:
        pd.DataFrame: Data with handled missing values
    """
    df = df.copy()

    # Find customers who have missing plan_type values
    customers_with_missing = df[df['plan_type'].isna()]['customer_id'].unique()
    sliced_df = df[df['customer_id'].isin(customers_with_missing)]
    # Only process customers who have missing values
    for customer_id, customer_data in sliced_df.groupby('customer_id'):
        # Find where plan_type is missing for this customer
        missing_mask = customer_data['plan_type'].isna()
        missing_indices = customer_data.index[missing_mask]
        # For each missing value
        for idx in missing_indices:
            # Find position of this record within the customer's timeline
            customer_indices = customer_data.index
            pos = customer_indices.get_loc(idx)
            
            if pos == 0:
                df.loc[idx, 'plan_type'] = 'Basic'
            else:
                # Get the previous value from the same customer
                prev_idx = customer_indices[pos - 1]
                df.loc[idx, 'plan_type'] = df.loc[prev_idx, 'plan_type']
    
    
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
    # Check for duplicates
    duplicates = df.duplicated(subset=subset, keep='first')
    n_duplicates = duplicates.sum()
    
    if n_duplicates > 0:
        print(f"Found {n_duplicates} duplicate rows{' based on ' + ', '.join(subset) if subset else ''}")
    else:
        print("No duplicates found" + (f" based on {', '.join(subset)}" if subset else ""))
    
    return df.drop_duplicates(subset=subset)


def add_binary_columns_for_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add binary columns to indicate missing values.
    
    Args:
        df (pd.DataFrame): Input data
        
    Returns:
        pd.DataFrame: Data with binary columns for missing values
    """
    df = df.copy()
    
    df['is_missing_plan_type'] = df['plan_type'].isnull().astype(int)
    df['is_missing_transaction_amount'] = df['transaction_amount'].isnull().astype(int)

    return df

