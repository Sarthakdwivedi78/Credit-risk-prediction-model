import pandas as pd
import json

# --- 1. DEFINE THE FINAL TRAINING DATASET ---
training_file = 'training_dataset.xlsx'

# This is the full list of columns your model expects.
MODEL_COLUMNS = [
    'pct_tl_open_L6M', 'pct_tl_closed_L6M', 'Tot_TL_closed_L12M', 'pct_tl_closed_L12M',
    'Tot_Missed_Pmnt', 'CC_TL', 'Home_TL', 'PL_TL', 'Secured_TL', 'Unsecured_TL',
    'Other_TL', 'Age_Oldest_TL', 'Age_Newest_TL', 'time_since_recent_payment',
    'max_recent_level_of_deliq', 'num_deliq_6_12mts', 'num_times_60p_dpd',
    'num_std_12mts', 'num_sub', 'num_sub_6mts', 'num_sub_12mts', 'num_dbt',
    'num_dbt_12mts', 'num_lss', 'recent_level_of_deliq', 'CC_enq_L12m',
    'PL_enq_L12m', 'time_since_recent_enq', 'enq_L3m', 'NETMONTHLYINCOME',
    'Time_With_Curr_Empr', 'CC_Flag', 'PL_Flag', 'pct_PL_enq_L6m_of_ever',
    'pct_CC_enq_L6m_of_ever', 'HL_Flag', 'GL_Flag', 'EDUCATION',
    'MARITALSTATUS_Married', 'MARITALSTATUS_Single', 'GENDER_F', 'GENDER_M',
    'last_prod_enq2_AL', 'last_prod_enq2_CC', 'last_prod_enq2_ConsumerLoan',
    'last_prod_enq2_HL', 'last_prod_enq2_PL', 'last_prod_enq2_others',
    'first_prod_enq2_AL', 'first_prod_enq2_CC', 'first_prod_enq2_ConsumerLoan',
    'first_prod_enq2_HL', 'first_prod_enq2_PL', 'first_prod_enq2_others'
]


# --- 2. LOAD THE DATA ---
print(f"Loading the final training dataset: {training_file}...")
try:
    df = pd.read_excel(training_file)
    print("Dataset loaded successfully.")
    
except FileNotFoundError:
    print(f"Error: The file '{training_file}' was not found. Make sure it's in the same folder.")
    exit()
except Exception as e:
    print(f"An error occurred: {e}. Make sure you have 'openpyxl' installed.")
    exit()

# --- 3. CALCULATE STATISTICS AND EXTRACT SAMPLES ---
print("Calculating statistics and extracting samples...")
output_data = {"column_stats": {}, "sample_cases": {}}

# Calculate stats for each column
for col in MODEL_COLUMNS:
    if col in df.columns:
        if df[col].dtype == 'bool' or df[col].nunique(dropna=True) <= 2 and df[col].min() == 0 and df[col].max() == 1:
            output_data["column_stats"][col] = {"type": "bool", "hint": "Valid inputs: 0 (No) or 1 (Yes)."}
        else:
            min_val = df[col].dropna().min()
            max_val = df[col].dropna().max()
            output_data["column_stats"][col] = {
                "type": "numeric",
                "min": float(min_val), "max": float(max_val),
                "hint": f"Typical Range: {min_val} to {max_val}"
            }
    else:
        print(f"CRITICAL WARNING: The model column '{col}' was not found in the training dataset!")

# Extract 5 sample rows for each 'Approved_Flag' category
if 'Approved_Flag' in df.columns:
    categories = ['P1', 'P2', 'P3', 'P4']
    for cat in categories:
        sample_df = df[df['Approved_Flag'] == cat].sample(n=min(5, len(df[df['Approved_Flag'] == cat])), random_state=1)
        # Convert sample to a list of dictionaries for JSON compatibility
        output_data["sample_cases"][cat] = sample_df[MODEL_COLUMNS].to_dict('records')
        print(f"Extracted {len(sample_df)} sample(s) for category {cat}.")
else:
    print("CRITICAL WARNING: 'Approved_Flag' column not found, cannot extract samples.")


# --- 4. SAVE DATA TO A FILE ---
output_filename = 'column_stats.json'
with open(output_filename, 'w') as f:
    json.dump(output_data, f, indent=4)

print(f"\nStatistics and sample cases have been saved to '{output_filename}'!")

