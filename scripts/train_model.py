"""
Script to train and save the churn prediction model.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_data, preprocess_data, split_data
from src.feature_engineering import FeatureEngineer
from src.model import ChurnModel
from src.evaluation import calculate_metrics, save_metrics
from src.utils import setup_directories
from src import config

def main():
    # Create necessary directories
    setup_directories(['models', 'results'])
    
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
    
    # Train model
    print("Training model...")
    model = ChurnModel()
    model.train(X_train, y_train)
    
    # Save model
    print("Saving model...")
    model.save()
    
    # Evaluate model
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    
    metrics = calculate_metrics(y_test, y_pred, y_prob)
    save_metrics(metrics)
    
    print("Training completed successfully!")
    print("Metrics:", metrics)

if __name__ == "__main__":
    main() 