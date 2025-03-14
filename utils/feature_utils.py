import pandas as pd
import numpy as np
from datetime import datetime

def generate_features(df):
    """
    Generate customer-level features from time-series data for churn prediction.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe with customer time-series data
        
    Returns:
    --------
    pandas.DataFrame
        Dataframe with customer-level features
    """
    # Convert date columns to datetime
    df['date'] = pd.to_datetime(df['date'])
    df['issuing_date'] = pd.to_datetime(df['issuing_date'])
    
    # Handle missing transaction_amount values
    df['transaction_amount'] = df['transaction_amount'].fillna(df.groupby('customer_id')['transaction_amount'].transform('mean'))
    
    # Create an empty list to store customer features
    customer_features = []
    
    # Process each customer separately
    for customer_id, customer_data in df.groupby('customer_id'):
        # Sort by date to ensure chronological order
        customer_data = customer_data.sort_values('date')
        
        # Get last record (December 2023)
        last_record = customer_data.iloc[-1]
        
        # Get second last record (November 2023)
        second_last_record = customer_data.iloc[-2] if len(customer_data) > 1 else None
        
        # === Basic Customer Information ===
        features = {
            'customer_id': customer_id,
            'last_plan_type': last_record['plan_type'],
            'is_churned_last_month': last_record['churn'],
        }
        
        # === Date-Dependent Features (at least 3) ===
        
        # 1. Customer Tenure (in months) as of December 2023
        reference_date = pd.to_datetime('2023-12-31')
        features['tenure_months'] = (reference_date - last_record['issuing_date']).days / 30
        
        # 2. Days since last plan change
        plan_changes = customer_data['plan_type'].ne(customer_data['plan_type'].shift()).astype(int)
        if plan_changes.sum() > 0:
            last_change_idx = customer_data.index[plan_changes.values == 1][-1] if plan_changes.sum() > 0 else customer_data.index[0]
            last_change_date = customer_data.loc[last_change_idx, 'date']
            features['days_since_plan_change'] = (reference_date - last_change_date).days
        else:
            features['days_since_plan_change'] = (reference_date - customer_data.iloc[0]['date']).days
        
        # 3. Month over month transaction amount change (recent trend)
        if second_last_record is not None:
            features['mom_transaction_change'] = (
                last_record['transaction_amount'] - second_last_record['transaction_amount']
            ) / (second_last_record['transaction_amount'] + 1e-10)  # Avoid division by zero
        else:
            features['mom_transaction_change'] = 0
            
        # 4. Quarterly transaction volatility (standard deviation over last 3 months)
        if len(customer_data) >= 3:
            features['recent_volatility'] = customer_data['transaction_amount'].iloc[-3:].std()
        else:
            features['recent_volatility'] = 0
            
        # 5. Average transaction amount
        features['avg_transaction_amount'] = customer_data['transaction_amount'].mean()
        
        # 6. Transaction trend over the year (slope of linear regression)
        if len(customer_data) > 1:
            # Create x values (months indexed from 0)
            x = np.arange(len(customer_data))
            # Get y values (transaction amounts)
            y = customer_data['transaction_amount'].values
            # Fit a linear regression line
            slope, _ = np.polyfit(x, y, 1)
            features['transaction_trend'] = slope
        else:
            features['transaction_trend'] = 0
        
        # === Plan Type Features ===
        
        # 7. Total number of plan changes
        features['total_plan_changes'] = plan_changes.sum()
        
        # 8. Percentage of months in each plan type
        plan_counts = customer_data['plan_type'].value_counts()
        total_months = len(customer_data)
        
        for plan in ['Basic', 'Standard', 'Premium']:
            features[f'pct_{plan.lower()}_plan'] = plan_counts.get(plan, 0) / total_months
        
        # 9. Last plan change direction (upgrade/downgrade/same)
        if len(customer_data) > 1:
            plan_hierarchy = {'Basic': 1, 'Standard': 2, 'Premium': 3}
            last_plan = plan_hierarchy.get(last_record['plan_type'], 0)
            prev_plan = plan_hierarchy.get(second_last_record['plan_type'], 0)
            
            if last_plan > prev_plan:
                features['last_plan_change_type'] = 1  # Upgrade
            elif last_plan < prev_plan:
                features['last_plan_change_type'] = -1  # Downgrade
            else:
                features['last_plan_change_type'] = 0  # No change
        else:
            features['last_plan_change_type'] = 0
            
        # === Transaction Pattern Features ===
        
        # 10. Number of months with missing transactions
        features['missing_transaction_months'] = customer_data['transaction_amount'].isnull().sum()
        
        # 11. Coefficient of variation (measures relative volatility)
        if customer_data['transaction_amount'].mean() > 0:
            features['transaction_cv'] = customer_data['transaction_amount'].std() / customer_data['transaction_amount'].mean()
        else:
            features['transaction_cv'] = 0
            
        # 12. Maximum transaction amount
        features['max_transaction_amount'] = customer_data['transaction_amount'].max()
        
        # 13. Minimum transaction amount
        features['min_transaction_amount'] = customer_data['transaction_amount'].min()
        
        # === Seasonal Pattern Features ===
        
        # 14. Quarter-based transaction patterns
        q1_data = customer_data[customer_data['date'].dt.month.isin([1, 2, 3])]
        q2_data = customer_data[customer_data['date'].dt.month.isin([4, 5, 6])]
        q3_data = customer_data[customer_data['date'].dt.month.isin([7, 8, 9])]
        q4_data = customer_data[customer_data['date'].dt.month.isin([10, 11, 12])]
        
        features['q1_avg_transaction'] = q1_data['transaction_amount'].mean() if len(q1_data) > 0 else 0
        features['q2_avg_transaction'] = q2_data['transaction_amount'].mean() if len(q2_data) > 0 else 0
        features['q3_avg_transaction'] = q3_data['transaction_amount'].mean() if len(q3_data) > 0 else 0
        features['q4_avg_transaction'] = q4_data['transaction_amount'].mean() if len(q4_data) > 0 else 0
        
        # 15. Recent vs historical spending ratio
        recent_6m = customer_data.iloc[-6:]['transaction_amount'].mean() if len(customer_data) >= 6 else customer_data['transaction_amount'].mean()
        earlier_6m = customer_data.iloc[:-6]['transaction_amount'].mean() if len(customer_data) > 6 else None
        
        if earlier_6m is not None and earlier_6m > 0:
            features['recent_to_historical_ratio'] = recent_6m / earlier_6m
        else:
            features['recent_to_historical_ratio'] = 1.0
            
        # Add the customer features to our list
        customer_features.append(features)
    
    # Convert the list of dictionaries to a DataFrame
    features_df = pd.DataFrame(customer_features)
    
    return features_df