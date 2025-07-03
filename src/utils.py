# utils.py

import pandas as pd
import numpy as np
import joblib
import os
import logging

# Logging config
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_csv(path):
    """
    Loads a CSV file and returns a DataFrame.
    """
    if not os.path.exists(path):
        logging.error(f"File not found: {path}")
        raise FileNotFoundError(f"File not found: {path}")
    
    logging.info(f"Loading dataset from {path}")
    return pd.read_csv(path)


def save_model(model, filename='churn_model.pkl'):
    """
    Saves a trained model to disk using joblib.
    """
    joblib.dump(model, filename)
    logging.info(f"✅ Model saved to {filename}")


def load_model(filename='churn_model.pkl'):
    """
    Loads a trained model from disk.
    """
    if not os.path.exists(filename):
        logging.error(f"Model file not found: {filename}")
        raise FileNotFoundError(f"Model file not found: {filename}")

    logging.info(f"🔁 Loading model from {filename}")
    return joblib.load(filename)


def print_model_summary(model, X_train):
    """
    Prints number of features and feature importances (if available).
    """
    print(f"Model: {model.__class__.__name__}")
    print(f"Number of features: {X_train.shape[1]}")
    
    if hasattr(model, 'feature_importances_'):
        print("Top 10 Feature Importances:")
        importances = model.feature_importances_
        sorted_idx = np.argsort(importances)[::-1]
        for idx in sorted_idx[:10]:
            print(f"Feature {idx}: {importances[idx]:.4f}")
    else:
        print("Feature importances not available for this model.")


def preprocess_new_input(input_data, scaler):
    """
    Scales new input data using the provided scaler.
    """
    input_scaled = scaler.transform([input_data])
    return input_scaled

