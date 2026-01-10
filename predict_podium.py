import pandas as pd
import sqlite3
import fastf1 as ff1
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

# --- 1. Load Data from Database ---
DB_FILE = "f1_project.db"
TABLE_NAME = "raw_results"

print("Loading data from database...")
try:
    conn = sqlite3.connect(DB_FILE)
    df_raw = pd.read_sql_query(f"SELECT * FROM {TABLE_NAME}", conn)
    conn.close()
    # Remove duplicates
    df_raw.drop_duplicates(subset=['Year', 'RaceName', 'FullName'], keep='last', inplace=True)
    
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

print(f"Loaded {len(df_raw)} results.")

# --- 2. Feature Engineering ---
print("Starting feature engineering...")

# 2a. Get Race Dates (to sort chronologically)
schedules = []
for year in df_raw['Year'].unique():
    try:
        schedule = ff1.get_event_schedule(year)
        schedule['Year'] = year
        schedules.append(schedule[['EventName', 'EventDate', 'Year']])
    except Exception as e:
        print(f"Warning: Could not fetch schedule for {year}: {e}")

all_schedules = pd.concat(schedules)
all_schedules.rename(columns={'EventName': 'RaceName'}, inplace=True)
all_schedules.drop_duplicates(subset=['RaceName', 'Year'], keep='last', inplace=True)

# Merge dates
df = pd.merge(df_raw, all_schedules, on=['RaceName', 'Year'], how='left')
df.sort_values(by='EventDate', inplace=True)

# 2b. Create Rolling Features (Driver Form)
# Shift(1) ensures we only use PAST data
print("Calculating Driver rolling features...")
driver_grouped = df.groupby('FullName')

df['driver_avg_pos_5'] = driver_grouped['Position'].shift(1).rolling(5, min_periods=1).mean()
df['driver_avg_grid_5'] = driver_grouped['GridPosition'].shift(1).rolling(5, min_periods=1).mean()
df['driver_sum_points_5'] = driver_grouped['Points'].shift(1).rolling(5, min_periods=1).sum()

# 2c. Create Team Rolling Features (Car Form) - NEW!
# This helps the model understand if the CAR is fast, even if the driver is new (like Hamilton at Ferrari)
print("Calculating Team rolling features...")
team_grouped = df.groupby('TeamName')

df['team_avg_pos_5'] = team_grouped['Position'].shift(1).rolling(5, min_periods=1).mean()
df['team_sum_points_5'] = team_grouped['Points'].shift(1).rolling(5, min_periods=1).sum()

# 2d. Create Target Variable
df['FinishedTop3'] = (df['Position'] <= 3).astype(int)

# --- 3. Preprocessing ---
print("Preprocessing data for modeling...")

# Define Features
features = [
    'GridPosition', 
    'driver_avg_pos_5', 
    'driver_avg_grid_5', 
    'driver_sum_points_5',
    'team_avg_pos_5',      # New
    'team_sum_points_5',   # New
    'TeamName',
    'RaceName'             # New: Capture track-specific characteristics
]
target = 'FinishedTop3'

# Drop rows where we don't have history (start of career) or valid grid position
df_model = df.dropna(subset=features)

# One-Hot Encoding for Teams and Race Names
X = pd.get_dummies(df_model[features], columns=['TeamName', 'RaceName'], drop_first=True)
y = df_model[target]

# --- 4. Split Data (Time Series Split) ---
print("Splitting data into Train (2020-2024) and Test (2025)...")

train_mask = df_model['Year'] < 2025
test_mask = df_model['Year'] == 2025

X_train = X[train_mask]
y_train = y[train_mask]
X_test = X[test_mask]
y_test = y[test_mask]

# --- CRITICAL FIX: Align Columns Correctly ---
# 1. Add missing columns to Test set (fill with 0)
missing_cols = set(X_train.columns) - set(X_test.columns)
for c in missing_cols:
    X_test[c] = 0

# 2. Ensure Test set has exactly the same columns in the same order as Train
X_test = X_test[X_train.columns]

print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

if X_test.empty:
    print("Error: No 2025 data found. Run the scraper script first.")
    exit()

# --- 5. Train Models ---

print("\n--- Training Model 1: Logistic Regression ---")
log_reg_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000))
])
log_reg_pipeline.fit(X_train, y_train)
log_reg_preds = log_reg_pipeline.predict(X_test)

print("\n--- Training Model 2: XGBClassifier ---")
# Scale_pos_weight handles class imbalance (Top 3 is rare)
ratio = float(np.sum(y_train == 0)) / np.sum(y_train == 1)

xgb_model = XGBClassifier(
    random_state=42, 
    eval_metric='logloss',
    scale_pos_weight=ratio,
    use_label_encoder=False
)
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)

# --- 6. Evaluate Models ---
print("\n--- Model Evaluation (on 2025 data) ---")
print("\nLogistic Regression Report:")
print(classification_report(y_test, log_reg_preds, target_names=['No_Top_3', 'Top_3']))

print("\nXGBoost Classifier Report:")
print(classification_report(y_test, xgb_preds, target_names=['No_Top_3', 'Top_3']))

# --- 7. Feature Importance Analysis (New!) ---
# Check what the model thinks matters most
feature_importances = pd.Series(xgb_model.feature_importances_, index=X_train.columns)
print("\n--- Top 5 Most Important Features (XGBoost) ---")
print(feature_importances.nlargest(5))

# --- 8. Universal Race Predictor (Past or Future) ---
print("\n--- Universal Race Predictor ---")

# CHANGE THIS to any race name you want
# Examples: "São Paulo Grand Prix" (Past) or "Las Vegas Grand Prix" (Future)
TARGET_RACE = "Las Vegas Grand Prix" 

print(f"Targeting: {TARGET_RACE}")

# CHECK: Does this race exist in our 2025 data?
# We check if the race name appears in the 2025 rows
race_exists = TARGET_RACE in df_model[df_model['Year'] == 2025]['RaceName'].values

if race_exists:
    # =========================================
    # SCENARIO A: PAST RACE (Use Real Data)
    # =========================================
    print(f"--> Status: PAST RACE (Data found). Using actual Grid Positions.")
    
    # 1. Get the date of the target race
    target_date = df_model[(df_model['RaceName'] == TARGET_RACE) & (df_model['Year'] == 2025)]['EventDate'].iloc[0]
    
    # 2. Train on everything BEFORE this date
    X_train_final = X[df_model['EventDate'] < target_date]
    y_train_final = y[df_model['EventDate'] < target_date]
    
    # 3. Test on THIS date
    X_test_final = X[df_model['EventDate'] == target_date]
    
    # Re-Train & Predict
    final_model = XGBClassifier(random_state=42, eval_metric='logloss', scale_pos_weight=ratio, use_label_encoder=False)
    final_model.fit(X_train_final, y_train_final)
    final_probs = final_model.predict_proba(X_test_final)[:, 1]

    # Show Results with ACTUAL Finish Position
    results = df_model[df_model['EventDate'] == target_date].copy()
    results['Predicted_Prob'] = final_probs
    results = results[['FullName', 'TeamName', 'GridPosition', 'Predicted_Prob', 'Position']]
    results.sort_values(by='Predicted_Prob', ascending=False, inplace=True)
    
    print("\n--- Results (Actual vs Predicted) ---")
    print(results.to_string(index=False))

else:
    # =========================================
    # SCENARIO B: FUTURE RACE (Simulate Data)
    # =========================================
    print(f"--> Status: FUTURE RACE (No data found). Simulating Grid Positions based on form.")

    # Get active drivers from the very last known race
    last_known_date = df_model['EventDate'].max()
    active_drivers = df_model[df_model['EventDate'] == last_known_date]['FullName'].unique()
    
    future_data = []
    for driver in active_drivers:
        driver_stats = df_model[df_model['FullName'] == driver].iloc[-1]
        
        # ESTIMATE GRID: Round their average grid position to the nearest whole number
        estimated_grid = round(driver_stats['driver_avg_grid_5'])
        estimated_grid = max(1, min(20, estimated_grid)) # Keep between 1 and 20

        row = {
            'FullName': driver,
            'TeamName': driver_stats['TeamName'],
            'GridPosition': estimated_grid, # Estimated!
            'driver_avg_pos_5': driver_stats['driver_avg_pos_5'],
            'driver_avg_grid_5': driver_stats['driver_avg_grid_5'],
            'driver_sum_points_5': driver_stats['driver_sum_points_5'],
            'team_avg_pos_5': driver_stats['team_avg_pos_5'],
            'team_sum_points_5': driver_stats['team_sum_points_5'],
            'RaceName': TARGET_RACE
        }
        future_data.append(row)
    
    df_future = pd.DataFrame(future_data)
    
    # Align Features
    X_future = pd.get_dummies(df_future[features], columns=['TeamName', 'RaceName'], drop_first=True)
    
    # Ensure columns match training data exactly
    missing_cols = set(X_train.columns) - set(X_future.columns)
    for c in missing_cols: X_future[c] = 0
    X_future = X_future[X_train.columns]

    # Predict
    future_probs = xgb_model.predict_proba(X_future)[:, 1]
    
    # Show Results (No "Position" column because race hasn't happened)
    df_future['Predicted_Top3_Prob'] = future_probs
    df_future = df_future[['FullName', 'TeamName', 'GridPosition', 'Predicted_Top3_Prob']]
    df_future.sort_values(by='Predicted_Top3_Prob', ascending=False, inplace=True)
    
    print("\n--- Prediction (Hypothetical Grid) ---")
    print(df_future.to_string(index=False))
