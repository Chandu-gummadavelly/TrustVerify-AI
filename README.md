# 🛡️ TrustVerify: Interpretable Credit Risk System
**End-to-End Machine Learning Pipeline with Explainable AI (XAI)**

###  Live Demo
https://trustverify-ai-ighpdgaqdcdjxw8xyisy96.streamlit.app/

###  Project Overview
TrustVerify is a financial risk assessment tool built to solve the "Black Box" problem in banking AI. It doesn't just predict loan defaults; it uses **LIME** to explain *why* a specific applicant was flagged as high-risk.

###  Tech Stack
- **Engine:** XGBoost (Gradient Boosting)
- **Interpretability:** LIME (Local Interpretable Model-agnostic Explanations)
- **Frontend:** Streamlit
- **Data:** Scikit-Learn Pipelines, Pandas, NumPy

###  Key Features
- **Class Imbalance Handling:** Used `scale_pos_weight` to address the 70/30 split in credit data.
- **XAI Integration:** Generates local feature importance plots for every prediction.
- **Production-Ready:** Built using modular Python scripts and professional Scikit-Learn pipelines.

###  How to Run Locally
1. Clone the repo.
2. `pip install -r requirements.txt`
3. `streamlit run app.py`
