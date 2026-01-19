import pandas as pd
import joblib
from lime.lime_tabular import LimeTabularExplainer
from preprocess import prepare_data

def explain_prediction():
    # 1. Load the Model and Data
    model = joblib.load('models/xgboost_model.pkl')
    X_train, X_test, y_train, y_test = prepare_data('data/german_credit_clean.csv')

    # 2. Extract the Preprocessor and the Classifier from the Pipeline
    # LIME needs to see the "transformed" data to explain it
    preprocessor = model.named_steps['preprocessor']
    classifier = model.named_steps['classifier']

    # Transform training data for the explainer
    X_train_transformed = preprocessor.transform(X_train)
    
    # 3. Initialize LIME Explainer
    explainer = LimeTabularExplainer(
        training_data=X_train_transformed,
        feature_names=preprocessor.get_feature_names_out(),
        class_names=['Good Risk', 'Bad Risk'],
        mode='classification'
    )

    # 4. Pick a specific "Bad Risk" case from the test set to explain
    # Let's pick the first instance where the model predicted '1' (Bad)
    X_test_transformed = preprocessor.transform(X_test)
    y_pred = classifier.predict(X_test_transformed)
    
    # Find the first index where prediction is 1
    sample_idx = (y_pred == 1).argmax()
    
    # 5. Generate Explanation
    print(f"🧐 Explaining prediction for Sample #{sample_idx}...")
    exp = explainer.explain_instance(
        data_row=X_test_transformed[sample_idx],
        predict_fn=classifier.predict_proba
    )

    # 6. Show the Result
    print("\n--- LIME Explanation ---")
    for feature, weight in exp.as_list():
        print(f"{feature}: {weight:.4f}")
    
    # Save the explanation as an HTML file (Recruiters love this!)
    exp.save_to_file('models/explanation.html')
    print("\n✅ Explanation saved as 'models/explanation.html'. Open it in your browser!")

if __name__ == "__main__":
    explain_prediction()