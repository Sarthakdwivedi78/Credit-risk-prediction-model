import streamlit as st
import pandas as pd
import joblib
import xgboost
import json
from collections import Counter

# --- Page Configuration ---
st.set_page_config(
    page_title="Credit Risk Prediction App",
    page_icon="🤖",
    layout="wide"
)

# --- Initialize Session State ---
if 'selected_sample_for_sidebar' not in st.session_state:
    st.session_state.selected_sample_for_sidebar = None
if 'staged_profiles' not in st.session_state:
    st.session_state.staged_profiles = [] # This will be our "shopping cart"

# --- Mappings and Constants ---
RISK_CATEGORIES = {
    0: {"label": "Good Risk (P1)", "color": "green"},
    1: {"label": "Low Risk (P2)", "color": "blue"},
    2: {"label": "High Risk (P3)", "color": "orange"},
    3: {"label": "Very High Risk (P4)", "color": "red"}
}

COLUMN_DESCRIPTIONS = {
    'pct_tl_open_L6M': "Percent of accounts opened in the last 6 months", 'pct_tl_closed_L6M': "Percent of accounts closed in the last 6 months",
    'Tot_TL_closed_L12M': "Total accounts closed in the last 12 months", 'pct_tl_closed_L12M': "Percent of accounts closed in the last 12 months",
    'Tot_Missed_Pmnt': "Total missed payments ever", 'CC_TL': "Number of Credit Card accounts", 'Home_TL': "Number of Home Loan accounts",
    'PL_TL': "Number of Personal Loan accounts", 'Secured_TL': "Number of secured loans", 'Unsecured_TL': "Number of unsecured loans",
    'Other_TL': "Number of other types of accounts", 'Age_Oldest_TL': "Age of oldest account (in months)", 'Age_Newest_TL': "Age of newest account (in months)",
    'time_since_recent_payment': "Days since the last payment was made", 'max_recent_level_of_deliq': "Max delinquency level in recent history",
    'num_deliq_6_12mts': "Number of late payments 6-12 months ago", 'num_times_60p_dpd': "Number of times 60+ days past due",
    'num_std_12mts': "Number of on-time payment accounts (last 12 months)", 'num_sub': "Number of accounts with late payments",
    'num_sub_6mts': "Number of late payment accounts (last 6 months)", 'num_sub_12mts': "Number of late payment accounts (last 12 months)",
    'num_dbt': "Number of 'doubtful' accounts", 'num_dbt_12mts': "Number of 'doubtful' accounts (last 12 months)", 'num_lss': "Number of 'loss' accounts",
    'recent_level_of_deliq': "Current delinquency level", 'CC_enq_L12m': "CC applications (last 12 months)", 'PL_enq_L12m': "PL applications (last 12 months)",
    'time_since_recent_enq': "Days since the last credit application", 'enq_L3m': "Credit applications (last 3 months)", 'NETMONTHLYINCOME': "Net monthly income",
    'Time_With_Curr_Empr': "Time with current employer (months)", 'CC_Flag': "Has a Credit Card? (1=Yes, 0=No)", 'PL_Flag': "Has a Personal Loan? (1=Yes, 0=No)",
    'pct_PL_enq_L6m_of_ever': "% of all-time PL applications in last 6 months", 'pct_CC_enq_L6m_of_ever': "% of all-time CC applications in last 6 months",
    'HL_Flag': "Has a Home Loan? (1=Yes, 0=No)", 'GL_Flag': "Has a Gold Loan? (1=Yes, 0=No)", 'EDUCATION': "Applicant's highest education level",
    'MARITALSTATUS_Married': "Is married? (1=Yes, 0=No)", 'MARITALSTATUS_Single': "Is single? (1=Yes, 0=No)",
    'GENDER_F': "Is female? (1=Yes, 0=No)", 'GENDER_M': "Is male? (1=Yes, 0=No)", 'last_prod_enq2_AL': "Last enquiry: Auto Loan? (1=Yes, 0=No)",
    'last_prod_enq2_CC': "Last enquiry: Credit Card? (1=Yes, 0=No)", 'last_prod_enq2_ConsumerLoan': "Last enquiry: Consumer Loan? (1=Yes, 0=No)",
    'last_prod_enq2_HL': "Last enquiry: Home Loan? (1=Yes, 0=No)", 'last_prod_enq2_PL': "Last enquiry: Personal Loan? (1=Yes, 0=No)",
    'last_prod_enq2_others': "Last enquiry: Other? (1=Yes, 0=No)", 'first_prod_enq2_AL': "First enquiry: Auto Loan? (1=Yes, 0=No)",
    'first_prod_enq2_CC': "First enquiry: Credit Card? (1=Yes, 0=No)", 'first_prod_enq2_ConsumerLoan': "First enquiry: Consumer Loan? (1=Yes, 0=No)",
    'first_prod_enq2_HL': "First enquiry: Home Loan? (1=Yes, 0=No)", 'first_prod_enq2_PL': "First enquiry: Personal Loan? (1=Yes, 0=No)",
    'first_prod_enq2_others': "First enquiry: Other? (1=Yes, 0=No)",
}

EDUCATION_MAP = {"SSC / Others": 1, "12th": 2, "Graduate / UG / Professional": 3, "Post-Graduate": 4}
EDUCATION_MAP_REVERSE = {v: k for k, v in EDUCATION_MAP.items()}
MODEL_COLUMNS = list(COLUMN_DESCRIPTIONS.keys())

# --- Load Model and Data ---
@st.cache_data
def load_data():
    try:
        model = joblib.load('xgboost_model.pkl')
        with open('column_stats.json', 'r') as f:
            loaded_data = json.load(f)
            stats = loaded_data.get("column_stats", {})
            sample_cases = loaded_data.get("sample_cases", {})
        return model, stats, sample_cases
    except FileNotFoundError:
        st.error("Error: Make sure 'xgboost_model.pkl' and 'column_stats.json' are in the directory. Run 'prepare_data.py' first.")
        st.stop()

model, stats, sample_cases = load_data()

# --- Pre-processing Function for File Upload ---
def preprocess_dataframe(df):
    df_processed = df.copy()
    if 'MARITALSTATUS' in df_processed.columns:
        df_processed = pd.concat([df_processed, pd.get_dummies(df_processed['MARITALSTATUS'], prefix='MARITALSTATUS').astype(int)], axis=1)
    if 'GENDER' in df_processed.columns:
        df_processed = pd.concat([df_processed, pd.get_dummies(df_processed['GENDER'], prefix='GENDER').astype(int)], axis=1)
    if 'last_prod_enq2' in df_processed.columns:
        df_processed = pd.concat([df_processed, pd.get_dummies(df_processed['last_prod_enq2'], prefix='last_prod_enq2').astype(int)], axis=1)
    if 'first_prod_enq2' in df_processed.columns:
        df_processed = pd.concat([df_processed, pd.get_dummies(df_processed['first_prod_enq2'], prefix='first_prod_enq2').astype(int)], axis=1)
    if 'EDUCATION' in df_processed.columns:
        education_mapping = {'SSC': 1, '12TH': 2, 'GRADUATE': 3, 'UNDER GRADUATE': 3, 'POST-GRADUATE': 4, 'OTHERS': 1, 'PROFESSIONAL': 3}
        df_processed['EDUCATION'] = df_processed['EDUCATION'].str.upper().map(education_mapping)
    for col in MODEL_COLUMNS:
        if col not in df_processed.columns:
            df_processed[col] = 0
    return df_processed[MODEL_COLUMNS]

# --- Sidebar ---
st.sidebar.title("Options")
input_method = st.sidebar.radio("Select Prediction Method", ("Manual Input", "File Upload"))

if input_method == "Manual Input":
    st.sidebar.header("Enter Applicant Details")
    def user_input_features():
        inputs = {}
        # If a sample is loaded, use it for defaults, otherwise use the overall stats
        source_data = st.session_state.selected_sample_for_sidebar if st.session_state.selected_sample_for_sidebar else stats
        for col in MODEL_COLUMNS:
            friendly_name = COLUMN_DESCRIPTIONS.get(col, col)
            col_stats = stats.get(col, {})
            stats_hint = col_stats.get('hint', 'No stats available.')
            
            if col == 'EDUCATION':
                default_code = source_data.get(col, 3) if st.session_state.selected_sample_for_sidebar else int(round(col_stats.get('mean', 3)))
                default_text = EDUCATION_MAP_REVERSE.get(default_code, "Graduate / UG / Professional")
                options = list(EDUCATION_MAP.keys())
                try: default_index = options.index(default_text)
                except ValueError: default_index = 2
                selected_education = st.sidebar.selectbox(f"{friendly_name}", options, index=default_index, help=stats_hint)
                inputs[col] = EDUCATION_MAP[selected_education]
            elif col_stats.get('type') == 'bool':
                default_bool = source_data.get(col, 0) if st.session_state.selected_sample_for_sidebar else 0
                inputs[col] = st.sidebar.selectbox(f"{friendly_name}", (0, 1), index=int(default_bool), help=stats_hint)
            else:
                default_value = source_data.get(col, 0.0) if st.session_state.selected_sample_for_sidebar else col_stats.get('min', 0.0)
                inputs[col] = st.sidebar.number_input(f"{friendly_name}", value=float(default_value), format="%.2f", help=stats_hint)
        return pd.DataFrame([inputs])
    input_df = user_input_features()

else: # File Upload
    st.sidebar.header("Upload Your Data File")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV or Excel file", type=['csv', 'xlsx'])
    st.sidebar.info("The file must contain raw columns (e.g., 'MARITALSTATUS') for pre-processing.")

# --- Main Page Display ---
st.title("Credit Risk Prediction App 💳")
st.write("Use the sidebar for single predictions and file uploads, or build a prediction batch from the samples below.")
st.write("---")

# --- MODIFIED: Single Prediction / File Upload Section is now FIRST ---
if input_method == "Manual Input":
    st.header("Single Applicant Prediction")
    st.write("The data entered in the sidebar is shown below.")
    st.dataframe(input_df.style.format("{:.2f}"))
    if st.button("Predict Manually Entered Risk"):
        try:
            prediction = model.predict(input_df)
            prediction_proba = model.predict_proba(input_df)
            predicted_class = prediction[0]
            result = RISK_CATEGORIES[predicted_class]
            st.markdown(f"The applicant is predicted to be: <span style='color:{result['color']}; font-size: 24px;'><b>{result['label']}</b></span>", unsafe_allow_html=True)
            st.write("Prediction Probabilities:")
            probabilities_df = pd.DataFrame([prediction_proba[0]], columns=[RISK_CATEGORIES[i]['label'] for i in range(len(RISK_CATEGORIES))])
            st.dataframe(probabilities_df.style.format("{:.2%}"))
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

elif input_method == "File Upload" and 'uploaded_file' in locals() and uploaded_file is not None:
    st.header("File Upload Prediction Results")
    try:
        raw_df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        st.write("Uploaded Raw Data:")
        st.dataframe(raw_df)
        processed_df = preprocess_dataframe(raw_df)
        predictions = model.predict(processed_df)
        results_df = raw_df.copy()
        results_df['Predicted Risk Category'] = [RISK_CATEGORIES[p]['label'] for p in predictions]
        st.write("Prediction Results:")
        st.dataframe(results_df)
    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")

st.write("---")

# --- Staging Area for Batch Prediction (Always Visible) ---
st.header("Interactive Prediction Batch")
if not st.session_state.staged_profiles:
    st.info("Your batch is empty. Add profiles from the 'Sample Applicant Profiles' section below.")
else:
    st.write(f"You have **{len(st.session_state.staged_profiles)}** profile(s) in your batch.")
    
    with st.expander("Click to view profiles in the batch"):
        staged_df = pd.DataFrame(st.session_state.staged_profiles)
        st.dataframe(staged_df)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Predict on Batch", type="primary"):
            staged_df = pd.DataFrame(st.session_state.staged_profiles)
            staged_df_ordered = staged_df[MODEL_COLUMNS]
            predictions = model.predict(staged_df_ordered)
            probabilities = model.predict_proba(staged_df_ordered)
            
            st.subheader("Batch Prediction Results")
            results_data = {
                'P1 (Good Risk) %': probabilities[:, 0],
                'P2 (Low Risk) %': probabilities[:, 1],
                'P3 (High Risk) %': probabilities[:, 2],
                'P4 (Very High Risk) %': probabilities[:, 3],
                'Predicted Risk': [RISK_CATEGORIES[p]['label'] for p in predictions]
            }
            results_df = pd.DataFrame(results_data)
            st.dataframe(results_df.style.format({
                'P1 (Good Risk) %': '{:.2%}', 'P2 (Low Risk) %': '{:.2%}',
                'P3 (High Risk) %': '{:.2%}', 'P4 (Very High Risk) %': '{:.2%}'
            }))
    with col2:
        if st.button("Clear Batch"):
            st.session_state.staged_profiles = []
            st.rerun()

st.write("---")

# --- Display Sample Cases ---
st.header("Sample Applicant Profiles by Risk Category")
st.info("Click 'Add to Batch' to stage a profile for group prediction, or 'View in Sidebar' to inspect it individually.")

if sample_cases:
    tab_titles = [f"{cat} Profiles" for cat in sample_cases.keys()]
    tabs = st.tabs(tab_titles)
    for i, category in enumerate(sample_cases.keys()):
        with tabs[i]:
            st.write(f"Showing examples of applicants categorized as **{category}**.")
            
            if st.button(f"Add All 5 '{category}' Samples to Batch", key=f"add_all_{category}"):
                for sample in sample_cases[category]:
                    if sample not in st.session_state.staged_profiles:
                        st.session_state.staged_profiles.append(sample)
                st.rerun()
            st.markdown("---")
            
            for idx, sample in enumerate(sample_cases[category]):
                with st.expander(f"Sample Profile {idx + 1} (from category {category})"):
                    s_col1, s_col2 = st.columns(2)
                    sample_items = list(sample.items())
                    mid_point = len(sample_items) // 2
                    with s_col1:
                        for key, value in sample_items[:mid_point]:
                            st.markdown(f"**{COLUMN_DESCRIPTIONS.get(key, key)}:** `{value}`")
                    with s_col2:
                        for key, value in sample_items[mid_point:]:
                            st.markdown(f"**{COLUMN_DESCRIPTIONS.get(key, key)}:** `{value}`")
                    
                    b_col1, b_col2 = st.columns(2)
                    with b_col1:
                        if st.button("Add to Prediction Batch", key=f"add_{category}_{idx}"):
                            st.session_state.staged_profiles.append(sample)
                            st.rerun()
                    with b_col2:
                        if st.button("View in Sidebar", key=f"load_{category}_{idx}"):
                            st.session_state.selected_sample_for_sidebar = sample
                            st.rerun()
else:
    st.warning("No sample cases were found in the `column_stats.json` file.")

# --- How to Use This App Section ---
st.write("---")
with st.expander("ℹ️ How to Use This App"):
    st.subheader("About This Application")
    st.markdown("""
    This is an advanced tool for predicting credit risk based on a multi-class XGBoost model. The model assesses loan applicants and assigns them to one of four risk categories, providing a detailed and nuanced evaluation beyond a simple "good" or "bad" prediction.
    """)

    st.subheader("Prediction Methods")
    st.markdown("""
    There are three ways to get a prediction:

    **1. Manual Input (Single Prediction):**
    - Select **"Manual Input"** in the sidebar.
    - A detailed form will appear with 54 fields. Fill in the applicant's data.
    - Hover over the **?** icon next to each field for guidance on typical data ranges.
    - Click the **"Predict Manually Entered Risk"** button on the main page to see the result for this single applicant.

    **2. File Upload (Bulk Prediction):**
    - Select **"File Upload"** in the sidebar.
    - Upload a CSV or Excel file containing one or more applicants. **Important:** The file must contain the raw, unprocessed data (e.g., a 'GENDER' column with 'M' or 'F', not 'GENDER_M').
    - The app will automatically pre-process the data and display a table with predictions for every row in your file.

    **3. Interactive Batch Prediction (Using Samples):**
    - This is the most powerful feature for analysis.
    - Scroll down to the **"Sample Applicant Profiles"** section.
    - Browse the profiles under the P1, P2, P3, and P4 tabs.
    - Click **"Add to Prediction Batch"** for any profile you want to analyze. Use the **"Add All 5 Samples"** button for quick bulk additions.
    - The selected profiles will appear in the **"Prediction Batch"** section at the top. Click the expander to view their data.
    - Click the **"Predict on Batch"** button to see a detailed, row-by-row breakdown of the risk probabilities for your selected group. Use the **"Clear Batch"** button to start over.
    """)

    st.subheader("Understanding the Risk Categories")
    st.markdown("""
    The model's prediction will fall into one of four categories:
    - **<span style='color:green'>Good Risk (P1):</span>** The highest quality applicants. Very low risk of default.
    - **<span style='color:blue'>Low Risk (P2):</span>** Good applicants who are generally reliable.
    - **<span style='color:orange'>High Risk (P3):</span>** Applicants with some negative indicators. Caution is advised.
    - **<span style='color:red'>Very High Risk (P4):</span>** Applicants with significant risk factors. High probability of default.
    """, unsafe_allow_html=True)

