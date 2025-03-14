"""
Model definition and training functions.
"""
import pickle
from typing import Tuple, Any, Dict, Optional
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from . import config
from . import model_config

class ChurnModel:
    def __init__(self, params: Optional[Dict] = None, config_name: Optional[str] = None):
        """
        Initialize the model with given parameters or a predefined configuration.
        
        Args:
            params (Dict, optional): Custom model parameters
            config_name (str, optional): Name of predefined configuration from EXPERIMENTAL_CONFIGS
        """
        if params is not None:
            self.params = params
        elif config_name is not None and config_name in model_config.EXPERIMENTAL_CONFIGS:
            self.params = model_config.EXPERIMENTAL_CONFIGS[config_name]
        else:
            self.params = model_config.DEFAULT_MODEL_PARAMS
            
        self.model = GradientBoostingClassifier(**self.params)
        
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
            pickle.dump(self, f)  # Save entire instance to preserve parameters
    
    @classmethod
    def load(cls, path: str = config.MODEL_PATH) -> 'ChurnModel':
        """
        Load a saved model.
        
        Args:
            path (str): Path to the saved model
            
        Returns:
            ChurnModel: Loaded model instance
        """
        with open(path, 'rb') as f:
            return pickle.load(f)  # Load entire instance
            
    def get_params(self) -> Dict:
        """
        Get the current model parameters.
        
        Returns:
            Dict: Current model parameters
        """
        return self.params 