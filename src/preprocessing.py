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

    df['date'] = pd.to_datetime(df['date'])
    df['issuing_date'] = pd.to_datetime(df['issuing_date'])
    df['plan_type'] = df['plan_type'].astype('category')
    df = remove_duplicates(df, subset=['customer_id', 'date'])
    df = add_binary_columns_for_missing_values(df)
    df = handle_missing_values(df)
    
    return df



def fill_missing_values_by_column(df: pd.DataFrame, column: str, ) -> pd.DataFrame:
    """
    Fill missing values in a specific column by using the previous value.
    
    Args:
        df (pd.DataFrame): Input data
    """
    
    df = df.copy()
    customers_with_missing = df[df[column].isna()]['customer_id'].unique()
    sliced_df = df[df['customer_id'].isin(customers_with_missing)]
    # Only process customers who have missing values
    for customer_id, customer_data in sliced_df.groupby('customer_id'):
        # Find where plan_type is missing for this customer
        missing_mask = customer_data[column].isna()
        missing_indices = customer_data.index[missing_mask]
        # For each missing value
        for idx in missing_indices:
            # Find position of this record within the customer's timeline
            customer_indices = customer_data.index
            pos = customer_indices.get_loc(idx)
            
            if pos == 0:
                if column == 'plan_type':
                    df.loc[idx, column] = 'Basic'
                else:
                    df.loc[idx, column] = customer_data[column].mean()
            else:
                # Get the previous value from the same customer
                prev_idx = customer_indices[pos - 1]
                df.loc[idx, column] = df.loc[prev_idx, column]

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

    df = fill_missing_values_by_column(df, 'plan_type')
    df = fill_missing_values_by_column(df, 'transaction_amount')
    
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


def split_to_features_and_label(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the data into features and label sets.
    """
    filter_dates = (df['date'] == '2023-11-01') | (df['date'] == '2023-12-01')
    df_features = df[~filter_dates]
    df_label = df[filter_dates]
    return df_features, df_label
    

def create_label_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a label column indicating churn status.
    """
    return df.groupby('customer_id').agg({'churn': 'max'}).reset_index()



def create_dummies(df: pd.DataFrame, column: str) -> pd.DataFrame:
    df = df.copy()
    dummies = pd.get_dummies(df[column], prefix=column)
    df = pd.concat([df, dummies], axis=1)
    df.drop(columns=[column], inplace=True)
    return df


