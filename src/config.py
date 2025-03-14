"""
Configuration parameters for the customer churn classification project.
"""

# Data paths
RAW_DATA_PATH = "data/raw/churn_data.csv"
PROCESSED_DATA_PATH = "data/processed/churn_data_processed.csv"
PREDICTED_DATA_PATH = "data/processed/churn_data_with_predictions.csv"
MODEL_PATH = "models/churn_model.pkl"
RESULTS_PATH = "results/metrics.json"

# Data splitting parameters
RANDOM_STATE = 111
TEST_SIZE = 0.2
TARGET_COLUMN = "Churn"

# Feature engineering parameters
CATEGORICAL_FEATURES = []  # Add categorical feature names
NUMERICAL_FEATURES = []    # Add numerical feature names

# Logging configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Results configuration
SAVE_PREDICTIONS = True
SAVE_FEATURE_IMPORTANCE = True
SAVE_MODEL_METRICS = True

# Model hyperparameters
MODEL_PARAMS = {
    "n_estimators": 100,
    "max_depth": 5,
    "learning_rate": 0.1
} 