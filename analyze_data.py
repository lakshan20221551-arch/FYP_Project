import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def analyze():
    print("Loading dataset...")
    try:
        df = pd.read_csv("data.csv", sep=';')
    except Exception as e:
        print(f"Error reading csv: {e}")
        return

    print("Original Columns:", df.columns.tolist())
    
    # Clean column names (strip whitespace and potential tabs)
    df.columns = [c.strip() for c in df.columns]
    print("\nCleaned Columns:", df.columns.tolist())

    # Encode Target
    # Dropout=0, Enrolled=1, Graduate=2 (This is the standard order for LabelEncoder sorted alphabetically? D, E, G)
    # Let's verify standard sorting: Dropout, Enrolled, Graduate.
    le = LabelEncoder()
    df['Target_Encoded'] = le.fit_transform(df['Target'])
    print("\nTarget Mapping:", dict(zip(le.classes_, range(len(le.classes_)))))

    # Calculate correlations
    # We only care about numerical correlations with 'Target_Encoded'
    corr = df.corr(numeric_only=True)['Target_Encoded'].sort_values(ascending=False)
    
    print("\nCorrelations with Target (Positive values indicate link to Graduation):")
    print(corr)

    # Filter positive correlations (excluding Target_Encoded itself)
    pos_corr = corr[corr > 0].drop('Target_Encoded', errors='ignore')
    print("\nPositive Correlation Columns:")
    for col, val in pos_corr.items():
        print(f"{col}: {val:.4f}")

if __name__ == "__main__":
    analyze()
