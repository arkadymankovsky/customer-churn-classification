"""
Script to train and save the churn prediction model.
"""
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_data
from src.preprocessing import preprocess_data, split_to_features_and_label, create_label_column, create_dummies
from src.data_splitting import split_data
from src.feature_engineering import generate_all_features
from src.model import ChurnModel, ChurnModelOptimizer
from src.evaluation import calculate_metrics, save_metrics
from src.utils import save_processed_data
from src import config

def main():    
    # Load and preprocess data
    print("Loading data...")
    df = load_data()
    print("Preprocessing data...")
    df_processed = preprocess_data(df)
    
    # Save processed data
    print("Saving processed data...")
    save_processed_data(df_processed, config.PROCESSED_DATA_PATH)

    # create features and label
    df_features, df_label = split_to_features_and_label(df_processed)
    print("Generating new features and aggregating by customer...")
    df_enriched = generate_all_features(df_features)
    df_label = create_label_column(df_label)
    df_final = pd.merge(df_enriched, df_label, on='customer_id', how='left')
    df_final = create_dummies(df_final, 'last_plan_type')


    # Save final data
    print("Saving enriched data...")
    save_processed_data(df_final, config.ENRICHED_DATA_PATH)


    # Split data
    print("Splitting data...")
    X_train, X_test, y_train, y_test = split_data(df_final)
    

    # 1. First, create and run the optimizer
    optimizer = ChurnModelOptimizer(
        model_type='catboost',
        metric='roc_auc',
        n_cv_folds=5
    )

    # Run optimization to find best parameters
    best_params = optimizer.optimize(
        X_train=X_train,
        y_train=y_train,
        n_trials=100,  # number of trials for optimization
        timeout=3600   # optional: 1 hour timeout
    )

    # 2. Create and train the model with the optimized parameters
    model = ChurnModel(
        model_type='catboost',  # same as optimizer
        params=best_params
    )

    # Train the model (includes cross-validation)
    metrics = model.train(X_train, y_train)

    # 3. (Optional) Make predictions and save the model
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    # Save the model
    model.save()

    # Print results
    print("Best parameters:", best_params)
    print("Cross-validation metrics:", metrics)
    # model.save()
    
    # # Evaluate model
    # print("Evaluating model...")
    # y_pred = model.predict(X_test)
    # y_prob = model.predict_proba(X_test)
    
    # metrics = calculate_metrics(y_test, y_pred, y_prob)
    # save_metrics(metrics)
    
    # print("Training completed successfully!")
    # print("Metrics:", metrics)

if __name__ == "__main__":
    main() 