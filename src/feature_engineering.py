"""
Feature creation and transformation functions.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from . import config
from datetime import datetime

def calculate_tenure_months(df):
    """
    Calculate tenure in months for each customer as of the last date in the dataset.
    
    Args:
        df: DataFrame with customer_id, date, and issuing_date columns
    
    Returns:
        DataFrame with customer_id and tenure_months
    """
    df = df.copy()
    # Get the maximum date in the dataset as reference point
    max_date = df['date'].max()
    
    # Get the earliest issuing_date for each customer
    customer_start_dates = df.groupby('customer_id')['issuing_date'].min().reset_index()
    customer_start_dates['issuing_date'] = pd.to_datetime(customer_start_dates['issuing_date'])
    
    # Calculate tenure in months
    customer_start_dates['tenure_months'] = ((max_date - customer_start_dates['issuing_date']).dt.days / 30).round(1)
    
    return customer_start_dates[['customer_id', 'tenure_months']]


def get_last_month_features(df):
    """
    Extract features from the last month of data for each customer.
    
    Args:
        df: DataFrame with customer data
    
    Returns:
        DataFrame with customer_id, last_plan_type, and is_churned_last_month
    """
    df = df.copy()

    # Get the last record for each customer
    last_records = df.sort_values('date').groupby('customer_id').last().reset_index()
    
    # Extract relevant features
    last_month_features = last_records[['customer_id', 'plan_type', 'churn']].copy()
    
    # Rename columns
    last_month_features.rename(columns={
        'plan_type': 'last_plan_type',
        'churn': 'is_churned_last_month'
    }, inplace=True)
    
    return last_month_features


def calculate_days_since_plan_change(df):
    """
    Calculate days since the last plan change for each customer.
    
    Args:
        df: DataFrame with customer_id, date, and plan_type columns
    
    Returns:
        DataFrame with customer_id and days_since_plan_change
    """
    df = df.copy()

    # Get the maximum date in the dataset
    max_date = df['date'].max()
    
    # Create a temporary DataFrame to track plan changes
    temp_df = df.sort_values(['customer_id', 'date']).copy()
    
    # Identify when plan type changes
    temp_df['plan_changed'] = temp_df.groupby('customer_id')['plan_type'].shift(1) != temp_df['plan_type']
    
    # For each customer, get the date of the most recent plan change
    plan_changes = temp_df[temp_df['plan_changed'] == True].groupby('customer_id').last().reset_index()
    
    if len(plan_changes) > 0:
        # Calculate days since last plan change
        plan_changes['days_since_plan_change'] = (max_date - plan_changes['date']).dt.days
    else:
        # If no plan changes, create empty DataFrame with required columns
        plan_changes = pd.DataFrame(columns=['customer_id', 'days_since_plan_change'])
    
    # Handle customers with no plan changes (use their first record date)
    customers_without_changes = set(df['customer_id'].unique()) - set(plan_changes['customer_id'])
    
    if customers_without_changes:
        first_records = df[df['customer_id'].isin(customers_without_changes)].groupby('customer_id').first().reset_index()
        first_records['days_since_plan_change'] = (max_date - first_records['date']).dt.days
        
        # Combine with customers who had plan changes
        plan_changes = pd.concat([
            plan_changes[['customer_id', 'days_since_plan_change']],
            first_records[['customer_id', 'days_since_plan_change']]
        ])
    
    return plan_changes[['customer_id', 'days_since_plan_change']]


def calculate_mom_transaction_change(df):
    """
    Calculate month-over-month transaction amount change for the last two months.
    
    Args:
        df: DataFrame with customer_id, date, and transaction_amount columns
    
    Returns:
        DataFrame with customer_id and mom_transaction_change
    """
    df = df.copy()
    # Sort by customer and date
    temp_df = df.sort_values(['customer_id', 'date']).copy()
    
    # Get the last two records for each customer
    last_two_months = temp_df.groupby('customer_id').tail(2).copy()
    
    # Create a pivot to get last and second-to-last month transactions
    pivot = last_two_months.pivot_table(
        index='customer_id',
        columns=last_two_months.groupby('customer_id').cumcount(ascending=False),
        values='transaction_amount'
    ).reset_index()
    
    # Rename columns for clarity
    if 0 in pivot.columns and 1 in pivot.columns:
        pivot.columns = ['customer_id', 'last_month_amount', 'previous_month_amount']
        
        # Calculate month-over-month change
        pivot['mom_transaction_change'] = (
            (pivot['last_month_amount'] - pivot['previous_month_amount']) / 
            pivot['previous_month_amount'].replace(0, np.nan)
        ).fillna(0)
    else:
        # Handle cases where customers might have only one month of data
        pivot['mom_transaction_change'] = 0
    
    return pivot[['customer_id', 'mom_transaction_change']]


def calculate_transaction_statistics(df):
    """
    Calculate various transaction amount statistics for each customer.
    
    Args:
        df: DataFrame with customer_id and transaction_amount columns
    
    Returns:
        DataFrame with customer_id and transaction statistics
    """
    df = df.copy()
    # Group by customer_id and calculate statistics
    transaction_stats = df.groupby('customer_id').agg({
        'transaction_amount': [
            ('avg_transaction_amount', 'mean'),
            ('max_transaction_amount', 'max'),
            ('min_transaction_amount', 'min'),
            ('transaction_cv', lambda x: x.std() / x.mean() if x.mean() > 0 else 0)
        ]
    }).reset_index()
    
    # Flatten multi-level column names
    transaction_stats.columns = [
        col[0] if col[1] == '' else col[1] 
        for col in transaction_stats.columns
    ]
    
    return transaction_stats


def calculate_plan_type_percentages(df):
    """
    Calculate percentage of time each customer spent on different plan types.
    
    Args:
        df: DataFrame with customer_id and plan_type columns
    
    Returns:
        DataFrame with customer_id and plan type percentages
    """
    df = df.copy()
    # Count occurrences of each plan type for each customer
    plan_counts = pd.crosstab(df['customer_id'], df['plan_type'], normalize='index')
    
    # Ensure all plan types are represented
    for plan_type in ['Basic', 'Standard', 'Premium']:
        if plan_type not in plan_counts.columns:
            plan_counts[plan_type] = 0
    
    # Select and rename only the relevant columns
    plan_percentages = plan_counts[['Standard', 'Premium']].copy()
    plan_percentages.columns = ['pct_standard_plan', 'pct_premium_plan']
    
    # Reset index to make customer_id a column
    plan_percentages.reset_index(inplace=True)
    
    return plan_percentages


def determine_last_plan_change_type(df):
    """
    Determine the direction of the last plan change for each customer.
    
    Args:
        df: DataFrame with customer_id, date, and plan_type columns
    
    Returns:
        DataFrame with customer_id and last_plan_change_type
    """
    df = df.copy()
    # Define plan hierarchy
    plan_hierarchy = {'Basic': 1, 'Standard': 2, 'Premium': 3}
    
    # Convert plan types to numeric values - explicitly convert from category to string first
    df['plan_numeric'] = df['plan_type'].astype(str).map(plan_hierarchy)
    
    # Sort by customer and date
    temp_df = df.sort_values(['customer_id', 'date']).copy()
    
    # Calculate plan change direction
    temp_df['prev_plan_numeric'] = temp_df.groupby('customer_id')['plan_numeric'].shift(1)
    temp_df['plan_change_type'] = temp_df['plan_numeric'] - temp_df['prev_plan_numeric']
    
    # Get the last plan change for each customer
    customer_last_changes = []
    
    for customer_id, customer_data in temp_df.groupby('customer_id'):
        # Find rows where plan changed
        plan_changes = customer_data[customer_data['plan_change_type'].notna() & (customer_data['plan_change_type'] != 0)]
        
        if len(plan_changes) > 0:
            # Get the most recent change
            last_change = plan_changes.iloc[-1]
            
            # Determine change type (1: upgrade, -1: downgrade, 0: no change)
            if last_change['plan_change_type'] > 0:
                change_type = 1  # Upgrade
            else:
                change_type = -1  # Downgrade
        else:
            # No plan changes for this customer
            change_type = 0
        
        customer_last_changes.append({
            'customer_id': customer_id,
            'last_plan_change_type': change_type
        })
    
    return pd.DataFrame(customer_last_changes)


def count_missing_values(df):
    """
    Count missing transaction amounts and plan types for each customer.
    
    Args:
        df: DataFrame with customer_id, is_missing_transaction_amount, and is_missing_plan_type columns
    
    Returns:
        DataFrame with customer_id, missing_transaction_months, and missing_plan_type_months
    """
    df = df.copy()
    # Count missing values for each customer
    missing_counts = df.groupby('customer_id').agg({
        'is_missing_transaction_amount': sum,
        'is_missing_plan_type': sum
    }).reset_index()
    
    # Rename columns
    missing_counts.rename(columns={
        'is_missing_transaction_amount': 'missing_transaction_months',
        'is_missing_plan_type': 'missing_plan_type_months'
    }, inplace=True)
    
    return missing_counts


def generate_all_features(df):
    """
    Main function to generate all required features.
    
    Args:
        df: Input DataFrame with the raw data
    
    Returns:
        DataFrame with all customer-level features
    """
    df = df.copy()

    # Generate all features
    tenure_features = calculate_tenure_months(df)
    last_month_features = get_last_month_features(df)
    plan_change_days = calculate_days_since_plan_change(df)
    mom_change = calculate_mom_transaction_change(df)
    transaction_stats = calculate_transaction_statistics(df)
    plan_percentages = calculate_plan_type_percentages(df)
    last_change_type = determine_last_plan_change_type(df)
    missing_counts = count_missing_values(df)
    
    
    # Combine all features
    # Start with customer_id as the base
    all_customers = df[['customer_id']].drop_duplicates()
    
    # Merge all feature dataframes
    features = all_customers
    for feature_df in [
        tenure_features, last_month_features, plan_change_days, mom_change,
        transaction_stats, plan_percentages, last_change_type, missing_counts
    ]:
        features = features.merge(feature_df, on='customer_id', how='left')
    
    return features