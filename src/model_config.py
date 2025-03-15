"""
Model-specific configuration parameters.
"""

# Data splitting parameters
RANDOM_STATE = 111
TEST_SIZE = 0.2
TARGET_COLUMN = "churn"

# Feature engineering parameters
CATEGORICAL_FEATURES = []
NUMERICAL_FEATURES = ['tenure_months', 'is_churned_last_month',
       'days_since_plan_change', 'mom_transaction_change',
       'avg_transaction_amount', 'max_transaction_amount',
       'min_transaction_amount', 'transaction_cv', 'pct_standard_plan',
       'pct_premium_plan', 'last_plan_change_type',
       'missing_transaction_months', 'missing_plan_type_months',
       'last_plan_type_Basic', 'last_plan_type_Premium',
       'last_plan_type_Standard']


# Default model hyperparameters
DEFAULT_MODEL_PARAMS = {
    "n_estimators": 100,
    "max_depth": 5,
    "learning_rate": 0.1,
    "random_state": 111
}

# Alternative model configurations for experimentation
EXPERIMENTAL_CONFIGS = {
    "baseline": {
        "n_estimators": 100,
        "max_depth": 3,
        "learning_rate": 0.1,
        "random_state": 111
    },
    "complex": {
        "n_estimators": 200,
        "max_depth": 7,
        "learning_rate": 0.05,
        "random_state": 111
    },
    "fast_learning": {
        "n_estimators": 50,
        "max_depth": 5,
        "learning_rate": 0.2,
        "random_state": 111
    }
}

# Cross-validation parameters
CV_PARAMS = {
    "n_splits": 5,
    "shuffle": True,
    "random_state": 111
}

# Model selection parameters
MODEL_SELECTION_PARAMS = {
    "scoring": ["accuracy", "precision", "recall", "f1", "roc_auc"],
    "refit": "f1"  # Primary metric for model selection
}

# Default parameters for CatBoost
DEFAULT_CATBOOST_PARAMS = {
    'iterations': 1000,
    'learning_rate': 0.1,
    'depth': 6,
    'loss_function': 'Logloss',
    'verbose': False
}

# Default parameters for XGBoost
DEFAULT_XGBOOST_PARAMS = {
    'n_estimators': 1000,
    'learning_rate': 0.1,
    'max_depth': 6,
    'objective': 'binary:logistic',
    'verbosity': 0
}

# Parameter spaces for optimization
CATBOOST_PARAM_SPACE = {
    'iterations': ('suggest_int', 100, 1000, {}),
    'learning_rate': ('suggest_float', 1e-3, 0.1, {'log': True}),
    'depth': ('suggest_int', 4, 10, {}),
    'l2_leaf_reg': ('suggest_float', 1e-8, 10.0, {'log': True}),
    'bootstrap_type': ('suggest_categorical', ['Bayesian', 'Bernoulli'], {}),
    'random_seed': 42  # Fixed parameter
}

XGBOOST_PARAM_SPACE = {
    'n_estimators': ('suggest_int', 100, 1000, {}),
    'max_depth': ('suggest_int', 4, 10, {}),
    'learning_rate': ('suggest_float', 1e-3, 0.1, {'log': True}),
    'min_child_weight': ('suggest_int', 1, 7, {}),
    'subsample': ('suggest_float', 0.6, 1.0, {}),
    'colsample_bytree': ('suggest_float', 0.6, 1.0, {}),
    'random_state': 42  # Fixed parameter
}

# Default parameters (used when not optimizing)
DEFAULT_CATBOOST_PARAMS = {
    'iterations': 500,
    'learning_rate': 0.1,
    'depth': 6,
    'l2_leaf_reg': 3.0,
    'bootstrap_type': 'Bayesian',
    'random_seed': 42
}

DEFAULT_XGBOOST_PARAMS = {
    'n_estimators': 500,
    'max_depth': 6,
    'learning_rate': 0.1,
    'min_child_weight': 3,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42
} 