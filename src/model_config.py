"""
Model-specific configuration parameters.
"""

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