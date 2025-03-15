"""
Model definition and training functions.
"""
import pickle
from typing import Tuple, Any, Dict, Optional, Literal, List
import pandas as pd
import numpy as np
import optuna
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from . import config
from . import model_config

class ChurnModel:
    def __init__(
        self, 
        model_type: Literal['catboost', 'xgboost'] = 'catboost',
        params: Optional[Dict] = None
    ):
        """
        Initialize the model with given parameters or a predefined configuration.
        
        Args:
            model_type (str): Type of model to use ('catboost' or 'xgboost')
            params (Dict, optional): Custom model parameters
        """
        self.model_type = model_type
        
        # Get parameters based on input
        if params is not None:
            self.params = params
        else:
            # Use default params based on model type
            if model_type == 'catboost':
                self.params = model_config.DEFAULT_CATBOOST_PARAMS
            else:  # xgboost
                self.params = model_config.DEFAULT_XGBOOST_PARAMS
        
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the model with current parameters."""
        if self.model_type == 'catboost':
            self.model = CatBoostClassifier(**self.params)
        else:  # xgboost
            self.model = XGBClassifier(**self.params)
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, float]:
        """
        Train the model with optional cross-validation.
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Dict with training metrics
        """
        X_train = X_train.copy()
        
        metrics = {}
        # Perform cross-validation
        cv_scores = self._cross_validate(X_train, y_train)
        metrics['cv_score_mean'] = cv_scores.mean()
        metrics['cv_score_std'] = cv_scores.std()
        
        # Train final model on full dataset
        if self.model_type == 'catboost':
            cat_features = X_train.select_dtypes(include=['category']).columns.tolist()
            self.model.fit(X_train, y_train, cat_features=cat_features)
        else:  # xgboost
            for col in X_train.select_dtypes(include=['category']).columns:
                X_train[col] = X_train[col].cat.codes
            self.model.fit(X_train, y_train)
        
        return metrics

    def _cross_validate(self, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> np.ndarray:
        """Perform cross-validation."""
        if self.model_type == 'xgboost':
            X = X.copy()
            for col in X.select_dtypes(include=['category']).columns:
                X[col] = X[col].cat.codes
        
        return cross_val_score(
            self.model, X, y, 
            cv=cv, 
            scoring='roc_auc',
            fit_params={'cat_features': X.select_dtypes(include=['category']).columns.tolist()} if self.model_type == 'catboost' else None
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data."""
        X = X.copy()
        if 'customer_id' in X.columns:
            X = X.drop('customer_id', axis=1)
        
        if self.model_type == 'xgboost':
            for col in X.select_dtypes(include=['category']).columns:
                X[col] = X[col].cat.codes
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get probability predictions."""
        X = X.copy()
        if 'customer_id' in X.columns:
            X = X.drop('customer_id', axis=1)
        
        if self.model_type == 'xgboost':
            for col in X.select_dtypes(include=['category']).columns:
                X[col] = X[col].cat.codes
        return self.model.predict_proba(X)
    
    def save(self, path: str = config.MODEL_PATH) -> None:
        """Save the model to disk."""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, path: str = config.MODEL_PATH) -> 'ChurnModel':
        """Load a saved model."""
        with open(path, 'rb') as f:
            return pickle.load(f)
            
    def get_params(self) -> Dict:
        """Get the current model parameters and type."""
        return {
            'model_type': self.model_type,
            'params': self.params
        }


class ChurnModelOptimizer:
    def __init__(
        self, 
        model_type: Literal['catboost', 'xgboost'] = 'catboost',
        metric: str = 'roc_auc',
        n_cv_folds: int = 5
    ):
        """
        Initialize the optimizer.
        
        Args:
            model_type: Type of model to optimize
            metric: Metric to optimize for
            n_cv_folds: Number of cross-validation folds
        """
        self.model_type = model_type
        self.metric = metric
        self.n_cv_folds = n_cv_folds

    def optimize(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series, 
        n_trials: int = 100,
        timeout: Optional[int] = None
    ) -> Dict:
        """
        Run optimization process.
        
        Args:
            X_train: Training features
            y_train: Training labels
            n_trials: Number of optimization trials
            timeout: Time limit in seconds
            
        Returns:
            Best parameters found
        """
        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: self._objective(trial, X_train, y_train),
            n_trials=n_trials,
            timeout=timeout
        )
        
        return study.best_params

    def _objective(self, trial: optuna.Trial, X: pd.DataFrame, y: pd.Series) -> float:
        """Optimization objective function."""
        params = self._suggest_params(trial)
        model = ChurnModel(model_type=self.model_type, params=params)
        
        # Perform cross-validation
        try:
            cv_scores = model._cross_validate(X, y, cv=self.n_cv_folds)
            return cv_scores.mean()
        except Exception as e:
            # Handle any errors during training
            print(f"Error during optimization: {str(e)}")
            return float('-inf')

    def _suggest_params(self, trial: optuna.Trial) -> Dict:
        """Get parameter suggestions from model_config."""
        param_space = (
            model_config.CATBOOST_PARAM_SPACE if self.model_type == 'catboost' 
            else model_config.XGBOOST_PARAM_SPACE
        )
        
        return {
            name: getattr(trial, func)(name, *args, **kwargs)
            for name, (func, *args, kwargs) in param_space.items()
        } 