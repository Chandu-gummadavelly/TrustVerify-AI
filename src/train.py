import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from preprocess import get_preprocessing_pipeline, prepare_data

def train_model():
    # 1. Load and Split Data
    print("📦 Loading data...")
    X_train, X_test, y_train, y_test = prepare_data('data/german_credit_clean.csv')

    # 2. Get Preprocessing Pipeline
    preprocessor = get_preprocessing_pipeline()

    # 3. Create the Full Pipeline (Preprocessor + XGBoost)
    # We use scale_pos_weight because our data is imbalanced (70/30 split)
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(
            n_estimators=100, 
            learning_rate=0.1, 
            max_depth=5, 
            scale_pos_weight=2.3, # Handles the 70/30 class imbalance
            random_state=42
        ))
    ])

    # 4. Train the Model
    print("🧠 Training XGBoost model...")
    model_pipeline.fit(X_train, y_train)

    # 5. Evaluate
    print("\n📊 Model Evaluation:")
    y_pred = model_pipeline.predict(X_test)
    print(classification_report(y_test, y_pred))

    # 6. Save the Model
    joblib.dump(model_pipeline, 'models/xgboost_model.pkl')
    print("\n✅ Model saved to models/xgboost_model.pkl")

if __name__ == "__main__":
    train_model()