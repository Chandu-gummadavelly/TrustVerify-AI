import streamlit as st
import pandas as pd
import joblib
import sys
import os

# 1. FIX THE MODULE PATH
# This finds the 'src' folder relative to where app.py is located
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

# Now we can import from preprocess safely
import preprocess 

st.set_page_config(page_title="TrustVerify: AI Credit Risk", layout="wide")

# 2. FIX THE MODEL PATH
@st.cache_resource
def load_assets():
    # Construct the absolute path to the model file
    model_path = os.path.join(current_dir, 'models', 'xgboost_model.pkl')
    
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}. Did you push the 'models' folder to GitHub?")
        return None, None
        
    model = joblib.load(model_path)
    
    # Load data for LIME initialization
    data_path = os.path.join(current_dir, 'data', 'german_credit_clean.csv')
    X_train, _, _, _ = preprocess.prepare_data(data_path)
    
    return model, X_train

model, X_train = load_assets()
# ... rest of your code
preprocessor = model.named_steps['preprocessor']
classifier = model.named_steps['classifier']

# 3. Sidebar for User Input
st.sidebar.header("📝 Applicant Details")

def user_input_features():
    # Adding the top 3 influential features we found
    duration = st.sidebar.slider("Loan Duration (months)", 4, 72, 24)
    amount = st.sidebar.number_input("Credit Amount (DM)", 250, 20000, 5000)
    age = st.sidebar.slider("Age", 18, 75, 30)
    
    # Simple versions of categorical inputs
    checking = st.sidebar.selectbox("Checking Account Status", ['<0 DM', '0<=X<200 DM', '>=200 DM', 'no checking'])
    savings = st.sidebar.selectbox("Savings Status", ['<100 DM', '100<=X<500 DM', '500<=X<1000 DM', '>=1000 DM', 'no savings'])
    
    # Constructing a dictionary (Simplified for this demo)
    # In a full project, you'd add all features here
    data = {
        'duration': duration, 'credit_amount': amount, 'age': age,
        'checking_status': checking, 'savings_status': savings,
        # Filling defaults for others to match our training shape
        'credit_history': 'existing paid', 'purpose': 'radio/tv', 'employment': '1<=X<4',
        'installment_commitment': 2, 'personal_status': 'male single', 'other_parties': 'none',
        'residence_since': 2, 'property_magnitude': 'real estate', 'other_payment_plans': 'none',
        'housing': 'own', 'existing_credits': 1, 'job': 'skilled', 'num_dependents': 1,
        'own_telephone': 'none', 'foreign_worker': 'yes'
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# 4. Prediction Logic
if st.button("Analyze Risk"):
    # Transform input
    input_transformed = preprocessor.transform(input_df)
    prediction = classifier.predict(input_transformed)[0]
    probability = classifier.predict_proba(input_transformed)[0]

    # Display Result
    col1, col2 = st.columns(2)
    with col1:
        if prediction == 0:
            st.success(f"✅ Result: LOW RISK (Prob: {probability[0]:.2f})")
        else:
            st.error(f"⚠️ Result: HIGH RISK (Prob: {probability[1]:.2f})")
    
    # 5. Explain with LIME
    with col2:
        st.write("🔍 **Why this decision?**")
        X_train_transformed = preprocessor.transform(X_train)
        explainer = LimeTabularExplainer(
            training_data=X_train_transformed,
            feature_names=preprocessor.get_feature_names_out(),
            class_names=['Good', 'Bad'],
            mode='classification'
        )
        exp = explainer.explain_instance(input_transformed[0], classifier.predict_proba)
        # Display LIME plot in Streamlit
        components.html(exp.as_html(), height=400, scrolling=True)