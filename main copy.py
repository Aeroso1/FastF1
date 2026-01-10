import fastf1 as ff1
import pandas as pd
import sqlite3
import time
import logging

# --- 0. Setup ---
# This will hide a lot of the 'INFO' and 'WARNING' messages from fastf1
logging.getLogger('fastf1').setLevel(logging.CRITICAL)

# Database file name
DB_FILE = "f1_project.db"

# Seasons we want to download
seasons_to_load = [2022, 2023, 2024, 2025]

# Connect to the SQLite database
# The file will be created if it doesn't exist
try:
    conn = sqlite3.connect(DB_FILE)
    print(f"Successfully connected to database: {DB_FILE}")
except Exception as e:
    print(f"Error connecting to database: {e}")
    exit()

# Enable the FastF1 cache
try:
    ff1.Cache.enable_cache('my_f1_cache')
    print("FastF1 cache enabled.")
except Exception as e:
    print(f"Error enabling cache: {e}")


# --- 1. Main Loop ---
def fetch_all_results():
    print("--- Starting Data Collection ---")
    
    for year in seasons_to_load:
        print(f"\nProcessing Season: {year}")
        
        # Get the full event schedule for the year
        schedule = ff1.get_event_schedule(year)
        
        # Loop over every event in the schedule
        # We use iterrows() to get the index and the row data
        for index, event in schedule.iterrows():
            
            # We only want to process events that have already happened
            if event['EventDate'] > pd.Timestamp.now():
                print(f"  > Skipping '{event['EventName']}' (not yet held)")
                continue

            print(f"  > Loading Race data for: {event['EventName']}")
            # --- 2. NEW: CHECK FOR AND LOAD SPRINT DATA ('S') ---
            if event['EventFormat'] == 'sprint':
                print(f"  > Sprint weekend detected. Loading Sprint data...")
                try:
                    # 'S' is the official identifier for a Sprint session
                    sprint_session = ff1.get_session(year, event['EventName'], 'S') 
                    sprint_session.load(telemetry=False, laps=False, weather=False, messages=False)
                    
                    if sprint_session.results is None:
                        print(f"  !! No results data for '{event['EventName']}' Sprint. Skipping.")
                        continue
                        
                    sprint_results = sprint_session.results
                    sprint_results['Year'] = year
                    sprint_results['RaceName'] = event['EventName']
                    sprint_results['SessionType'] = 'Sprint' # Mark this as Sprint points
                    
                    # Save sprint results to the SAME table
                    sprint_results.to_sql('raw_results', con=conn, if_exists='append', index=False)
                    
                except Exception as e:
                    print(f"  !! FAILED to load 'Sprint' for '{event['EventName']}': {e}")

            try:
                # Get the 'Race' session
                session = ff1.get_session(year, event['EventName'], 'R')
                
                # IMPORTANT: Load only the data we need (not telemetry!)
                session.load(telemetry=False, laps=False, weather=False, messages=False)
                
                # Get the results table (this is a pandas DataFrame)
                results = session.results
                
                # --- 2. Add Key Info ---
                # Add the 'Year' and 'RaceName' to each row
                # This is CRITICAL for our AI model later
                results['Year'] = year
                results['RaceName'] = event['EventName']
                
                # --- 3. Save to Database ---
                # 'raw_results' is the table name we chose in our project plan
                # 'if_exists='append'' adds the data to the table
                # 'index=False' stops pandas from saving the row number
                results.to_sql(
                    'raw_results',
                    con=conn,
                    if_exists='append',
                    index=False
                )
                
                # A small pause to be nice to the F1 servers
                time.sleep(1) 
                
            except Exception as e:
                # If a session fails (e.g., rained out, no data)
                print(f"  !! FAILED to load '{event['EventName']}': {e}")
    
    print("\n--- Data Collection Finished! ---")

# Run the function
fetch_all_results()

# Close the database connection
conn.close()
print("Database connection closed.")