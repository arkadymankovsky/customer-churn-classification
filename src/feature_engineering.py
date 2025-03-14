"""
Feature creation and transformation functions.
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from . import config

class FeatureEngineer:
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform the data with feature engineering.
        
        Args:
            df (pd.DataFrame): Input data
            
        Returns:
            pd.DataFrame: Transformed data
        """
        df = df.copy()
        
        # Scale numerical features
        for col in config.NUMERICAL_FEATURES:
            self.scalers[col] = StandardScaler()
            df[col] = self.scalers[col].fit_transform(df[[col]])
        
        # Encode categorical features
        for col in config.CATEGORICAL_FEATURES:
            self.encoders[col] = LabelEncoder()
            df[col] = self.encoders[col].fit_transform(df[col])
        
        return df
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted scalers and encoders.
        
        Args:
            df (pd.DataFrame): Input data
            
        Returns:
            pd.DataFrame: Transformed data
        """
        df = df.copy()
        
        # Transform numerical features
        for col in config.NUMERICAL_FEATURES:
            df[col] = self.scalers[col].transform(df[[col]])
        
        # Transform categorical features
        for col in config.CATEGORICAL_FEATURES:
            df[col] = self.encoders[col].transform(df[col])
        
        return df 