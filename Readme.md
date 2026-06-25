# 🛡️ Insurance Claim Settlement Bias Analytics Dashboard

An automated, interactive auditing tool built with Streamlit to detect, analyze, and visualize systemic biases in insurance claim approvals and repudiations. 

This dashboard allows claims officers and compliance teams to ingest historical claim data, perform diagnostic socio-economic and operational audits, and train robust Machine Learning models to evaluate decision stability.

## 📊 Key Features

* **Descriptive Analysis & Cross-Tabulation:** Interactive breakdown of claim approval rates across demographics, zones, and policy types.
* **Diagnostic Bias Audit:** Deep-dive visualizations exposing disparities across:
  * Socioeconomic status (Annual Income)
  * Demographic age groups
  * Regional teams/zones performance
  * Operational factors (Medical vs. Non-Medical, Payment Frequency)
* **Automated Machine Learning Pipeline:** Instant feature engineering (imputation, scaling, encoding) and supervised learning utilizing:
  * K-Nearest Neighbors (KNN)
  * Decision Trees
  * Random Forest
  * Gradient Boosting
* **Model Validation:** Auto-generated ROC curves and Confusion Matrices to evaluate predictive fairness and stability.

## 🛠️ Installation & Local Setup

To run this dashboard locally on your machine, follow these steps:

**1. Clone the repository or download the files:**
Ensure `app.py`, `Insurance.csv`, and `requirements.txt` are in the same folder.

**2. Create a virtual environment (Optional but recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate