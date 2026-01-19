import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def get_preprocessing_pipeline():
    """
    Creates a professional ColumnTransformer pipeline.
    """
    # Define our features based on the German Credit Dataset
    numeric_features = ['duration', 'credit_amount', 'installment_commitment', 'residence_since', 'age', 'existing_credits', 'num_dependents']
    
    categorical_features = [
        'checking_status', 'credit_history', 'purpose', 'savings_status', 
        'employment', 'personal_status', 'other_parties', 'property_magnitude',
        'other_payment_plans', 'housing', 'job', 'own_telephone', 'foreign_worker'
    ]

    # 1. Pipeline for Numbers: Fill missing (if any) and Scale
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    # 2. Pipeline for Categories: One-Hot Encode
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # 3. Combine them into a ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor

def prepare_data(file_path):
    df = pd.read_csv(file_path)
    X = df.drop('class', axis=1)
    y = df['class']
    
    # Stratified split ensures 'Bad' loans are evenly distributed in train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    print("Preprocess module ready.")