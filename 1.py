import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix

st.set_page_config(page_title="Insurance Claim Settlement Bias Dashboard", layout="wide")
st.title("🛡️ Insurance Claim Settlement Bias Analytics Dashboard")

# 1. Bulletproof Data Loader
@st.cache_data
def load_data(file_path_or_buffer):
    df = pd.read_csv(file_path_or_buffer)
    
    # Strip hidden whitespaces from column headers and make uppercase to prevent KeyErrors
    df.columns = df.columns.str.strip().str.upper()
    
    # Clean numeric format errors (commas to floats) safely
    if 'SUM_ASSURED' in df.columns and df['SUM_ASSURED'].dtype == 'object':
        df['SUM_ASSURED'] = df['SUM_ASSURED'].str.replace(',', '', regex=False).astype(float)
    if 'PI_ANNUAL_INCOME' in df.columns and df['PI_ANNUAL_INCOME'].dtype == 'object':
        df['PI_ANNUAL_INCOME'] = df['PI_ANNUAL_INCOME'].str.replace(',', '', regex=False).astype(float)
        
    return df

uploaded_file = st.sidebar.file_uploader("Upload Insurance CSV File", type=["csv"])

df = None
if uploaded_file is not None:
    df = load_data(uploaded_file)
else:
    try:
        df = load_data('Insurance.csv')
        st.sidebar.success("Loaded local sample 'Insurance.csv' data.")
    except Exception:
        st.sidebar.warning("Please upload the 'Insurance.csv' file to initialize the system.")

if df is not None:
    st.sidebar.header("Model Tuning Configurations")
    test_size = st.sidebar.slider("Test Partition Percentage (%)", 10, 50, 30, 5) / 100.0
    random_state = st.sidebar.number_input("Reproducibility Random Seed", value=42)

    # Ensure Target Column Exists
    if 'POLICY_STATUS' not in df.columns:
        st.error("CRITICAL ERROR: 'POLICY_STATUS' column not found in the dataset. Please check your CSV.")
        st.stop()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Descriptive Analysis", 
        "Diagnostic Bias Audit", 
        "Model Training Pipeline", 
        "Stability & Validation", 
        "Summary Audit Report"
    ])

    with tab1:
        st.header("Descriptive Cross-Tabulation Profiles")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Claim Profiles Assessed", df.shape[0])
            st.metric("Total Operational Dimensions", df.shape[1])
        with col2:
            status_counts = df['POLICY_STATUS'].value_counts()
            st.write("Target variable distributions (POLICY_STATUS):")
            st.dataframe(pd.DataFrame({'Count': status_counts, 'Percentage (%)': df['POLICY_STATUS'].value_counts(normalize=True)*100}))

        st.subheader("Interactive Feature Cross-Tabulation Matrix")
        # Safely filter available categorical columns
        possible_cats = ['PI_GENDER', 'EARLY_NON', 'MEDICAL_NONMED', 'PAYMENT_MODE', 'ZONE', 'PI_STATE']
        available_cats = [c for c in possible_cats if c in df.columns]
        
        if available_cats:
            selected_cat = st.selectbox("Select Dimension to Cross-Tabulate Against Status:", available_cats)

            ct_raw = pd.crosstab(df[selected_cat], df['POLICY_STATUS'])
            ct_percent = pd.crosstab(df[selected_cat], df['POLICY_STATUS'], normalize='index') * 100
            
            st.write("Raw Count:")
            st.dataframe(ct_raw)
            st.write("Percentage Distribution (%):")
            st.dataframe(ct_percent)

            fig, ax = plt.subplots(figsize=(10, 4.5))
            ct_percent.plot(kind='bar', stacked=True, color=['#2ec4b6', '#e71d36'], ax=ax)
            ax.set_ylabel("Proportional Rate (%)")
            ax.set_title(f"Settlement Outcomes Segmented by {selected_cat}")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        else:
            st.warning("No categorical columns found for cross-tabulation.")

    with tab2:
        st.header("Targeted Diagnostic Bias Investigations")
        diag_type = st.radio("Isolate Target Auditing Track:", ["Income-wise Bias", "Age-wise Bias", "Team/Zone Bias", "Operational Factors"])

        if diag_type == "Income-wise Bias":
            st.subheader("Socioeconomic Disparities Chart")
            if 'PI_ANNUAL_INCOME' in df.columns:
                fig, ax = plt.subplots(figsize=(10, 4.5))
                sns.boxplot(data=df, x='POLICY_STATUS', y='PI_ANNUAL_INCOME', palette=['#2ec4b6', '#e71d36'], ax=ax)
                ax.set_yscale('log')
                ax.set_title("Annual Income Log Distributions Across Settlement Labels")
                st.pyplot(fig)
                plt.close()

                df_bins = df.copy()
                df_bins['INCOME_BIN'] = pd.cut(df_bins['PI_ANNUAL_INCOME'], bins=[-1, 1, 100000, 300000, 500000, 100000000], labels=['No Income', 'Low (<100k)', 'Medium (100k-300k)', 'High (300k-500k)', 'Elite (>500k)'])
                
                st.write("Repudiation Probabilities Categorized by Income Tiers:")
                ct_inc = pd.crosstab(df_bins['INCOME_BIN'], df_bins['POLICY_STATUS'], normalize='index') * 100
                ct_inc.index = ct_inc.index.astype(str)
                st.dataframe(ct_inc)
            else:
                st.warning("Column 'PI_ANNUAL_INCOME' not found in dataset.")

        elif diag_type == "Age-wise Bias":
            st.subheader("Demographic Age Density Matrix")
            if 'PI_AGE' in df.columns:
                fig, ax = plt.subplots(figsize=(10, 4.5))
                sns.histplot(data=df, x='PI_AGE', hue='POLICY_STATUS', multiple='stack', palette=['#2ec4b6', '#e71d36'], bins=20, ax=ax)
                st.pyplot(fig)
                plt.close()

                df_bins = df.copy()
                df_bins['AGE_BIN'] = pd.cut(df_bins['PI_AGE'], bins=[0, 30, 45, 60, 75, 100])
                st.write("Repudiation Rate by Age Bracket:")
                ct_age = pd.crosstab(df_bins['AGE_BIN'], df_bins['POLICY_STATUS'], normalize='index') * 100
                ct_age.index = ct_age.index.astype(str)
                st.dataframe(ct_age)
            else:
                st.warning("Column 'PI_AGE' not found in dataset.")

        elif diag_type == "Team/Zone Bias":
            st.subheader("Regional Team Repudiation Vulnerability Matrix")
            if 'ZONE' in df.columns:
                zone_ct_norm = pd.crosstab(df['ZONE'], df['POLICY_STATUS'], normalize='index') * 100
                rep_rate = zone_ct_norm['Repudiate Death'] if 'Repudiate Death' in zone_ct_norm.columns else pd.Series(0, index=zone_ct_norm.index)