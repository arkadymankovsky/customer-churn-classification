"""
Script to evaluate model performance and generate explanations.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_data, preprocess_data, split_data
from src.feature_engineering import FeatureEngineer
from src.model import ChurnModel
from src.evaluation import calculate_metrics, save_metrics
from src.explainability import (
    calculate_shap_values,
    plot_feature_importance,
    get_feature_importance
)
from src import config

def main():
    # Load and preprocess data
    print("Loading data...")
    df = load_data()
    df = preprocess_data(df)
    
    # Split data
    print("Splitting data...")
    X_train, X_test, y_train, y_test = split_data(df)
    
    # Feature engineering
    print("Engineering features...")
    feature_engineer = FeatureEngineer()
    X_train = feature_engineer.fit_transform(X_train)
    X_test = feature_engineer.transform(X_test)
    
    # Load model
    print("Loading model...")
    model = ChurnModel.load()
    
    # Generate predictions
    print("Generating predictions...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    
    # Calculate metrics
    print("Calculating metrics...")
    metrics = calculate_metrics(y_test, y_pred, y_prob)
    save_metrics(metrics)
    
    # Generate SHAP explanations
    print("Generating SHAP values...")
    expected_value, shap_values = calculate_shap_values(model.model, X_test)
    
    # Plot feature importance
    print("Plotting feature importance...")
    plot_feature_importance(
        shap_values=shap_values,
        feature_names=X_test.columns,
        save_path='results/feature_importance.png'
    )
    
    # Get feature importance scores
    importance_df = get_feature_importance(shap_values, X_test.columns)
    importance_df.to_csv('results/feature_importance.csv', index=False)
    
    print("Evaluation completed successfully!")
    print("Metrics:", metrics)
    print("\nTop 5 most important features:")
    print(importance_df.head())

if __name__ == "__main__":
    main() 