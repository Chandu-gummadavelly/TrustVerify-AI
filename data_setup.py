import pandas as pd
import os

# 1. Define URL and Professional Column Names
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
COLUMNS = [
    "checking_status", "duration", "credit_history", "purpose", "credit_amount",
    "savings_status", "employment", "installment_commitment", "personal_status",
    "other_parties", "residence_since", "property_magnitude", "age",
    "other_payment_plans", "housing", "existing_credits", "job",
    "num_dependents", "own_telephone", "foreign_worker", "class"
]

def download_data():
    print(" Fetching raw data from UCI Repository...")
    # Read space-separated data
    df = pd.read_csv(DATA_URL, sep=' ', header=None, names=COLUMNS)
    
    # 2. Industry Standard: Re-map target variable
    # Original: 1 = Good, 2 = Bad. Standard: 0 = Good, 1 = Bad (Positive class is usually the 'event' or 'risk')
    df['class'] = df['class'].map({1: 0, 2: 1})
    
    # 3. Save to our data folder
    if not os.path.exists('data'):
        os.makedirs('data')
    
    df.to_csv('data/german_credit_clean.csv', index=False)
    print(" Data saved to data/german_credit_clean.csv")
    print(f"Dataset Shape: {df.shape}")
    print(df.head())

if __name__ == "__main__":
    download_data()