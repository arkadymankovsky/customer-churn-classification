"""
Script to generate predictions using the trained model.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_data, preprocess_data
from src.feature_engineering import FeatureEngineer
from src.model import ChurnModel
from src.utils import save_predictions
from src import config

def main():
    # Load and preprocess data
    print("Loading data...")
    df = load_data()
    df = preprocess_data(df)
    
    # Load model
    print("Loading model...")
    model = ChurnModel.load()
    
    # Feature engineering
    print("Engineering features...")
    feature_engineer = FeatureEngineer()
    X = feature_engineer.transform(df)
    
    # Generate predictions
    print("Generating predictions...")
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    
    # Save results
    print("Saving predictions...")
    save_predictions(
        df=df,
        predictions=predictions,
        probabilities=probabilities,
        output_path=config.PROCESSED_DATA_PATH
    )
    
    print("Predictions completed successfully!")

if __name__ == "__main__":
    main() 