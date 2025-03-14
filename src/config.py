"""
Configuration parameters for the customer churn classification project.
"""

# Data paths
RAW_DATA_PATH = "data/raw/churn_data.csv"
PROCESSED_DATA_PATH = "data/processed/churn_data_with_predictions.csv"
MODEL_PATH = "models/churn_model.pkl"
RESULTS_PATH = "results/metrics.json"

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_COLUMN = "Churn"

# Feature engineering parameters
CATEGORICAL_FEATURES = []  # Add categorical feature names
NUMERICAL_FEATURES = []    # Add numerical feature names

# Model hyperparameters
MODEL_PARAMS = {
    "n_estimators": 100,
    "max_depth": 5,
    "learning_rate": 0.1
} 