import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib
import os

def train_and_save_model():
    print("Loading dataset...")
    try:
        df = pd.read_csv("data.csv", sep=';')
    except FileNotFoundError:
        print("Error: data.csv not found. Please ensure the dataset is in the same directory.")
        return

    # Separate features and target
    X = df.drop("Target", axis=1)
    y = df["Target"]

    # Encode target
    # Dropout=0, Enrolled=1, Graduate=2
    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(y)
    
    # Save the target classes for later reference if needed
    print(f"Target classes: {target_encoder.classes_}")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Scale numeric data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    # We do not strictly need X_test for saving the model, but good for verification if we added metrics here.

    # Initialize model (Using XGBClassifier as shown in notebook imports, generally performs well)
    # Alternatively could use RandomForestClassifier
    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    
    print("Training XGBoost model...")
    model.fit(X_train, y_train)

    # Save artifacts
    print("Saving model and scaler...")
    joblib.dump(model, 'model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(target_encoder, 'target_encoder.pkl') # Also saving this just in case
    
    print("Done! 'model.pkl', 'scaler.pkl', and 'target_encoder.pkl' have been created.")

if __name__ == "__main__":
    train_and_save_model()
