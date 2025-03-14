"""
Model definition and training functions.
"""
import pickle
from typing import Tuple, Any
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from . import config

class ChurnModel:
    def __init__(self, params: dict = config.MODEL_PARAMS):
        self.model = GradientBoostingClassifier(**params)
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Train the model.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training labels
        """
        self.model.fit(X_train, y_train)
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X (pd.DataFrame): Input features
            
        Returns:
            np.ndarray: Predicted labels
        """
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get probability predictions.
        
        Args:
            X (pd.DataFrame): Input features
            
        Returns:
            np.ndarray: Predicted probabilities
        """
        return self.model.predict_proba(X)
    
    def save(self, path: str = config.MODEL_PATH) -> None:
        """
        Save the model to disk.
        
        Args:
            path (str): Path to save the model
        """
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
    
    @classmethod
    def load(cls, path: str = config.MODEL_PATH) -> 'ChurnModel':
        """
        Load a saved model.
        
        Args:
            path (str): Path to the saved model
            
        Returns:
            ChurnModel: Loaded model instance
        """
        instance = cls()
        with open(path, 'rb') as f:
            instance.model = pickle.load(f)
        return instance 