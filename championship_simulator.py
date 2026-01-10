import pandas as pd
import sqlite3
import fastf1 as ff1
import numpy as np
import warnings

# --- 1. CONFIGURATION ---
DB_FILE = "f1_project.db"
TABLE_NAME = "raw_results"
CURRENT_YEAR = 2025
SIMULATION_COUNT = 10000  # Run 10,000 simulated seasons
CONTENDERS = ['Max Verstappen', 'Lando Norris', 'Oscar Piastri']

# Suppress pandas warnings for cleaner output
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

def run_championship_simulation():
    """
    Loads all data, calculates current form, and simulates the
    rest of the season to predict the WDC.
    """
    
    # --- 2. LOAD DATA & GET CURRENT STANDINGS ---
    print("Loading all historical data...")
    try:
        conn = sqlite3.connect(DB_FILE)
        df_raw = pd.read_sql_query(f"SELECT * FROM {TABLE_NAME}", conn)
        conn.close()
        df_raw.drop_duplicates(subset=['Year', 'RaceName', 'FullName'], keep='last', inplace=True)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Filter for the current season
    df_2025 = df_raw[df_raw['Year'] == CURRENT_YEAR]
    
    # 2a. Get current points for all drivers
    current_standings = df_2025.groupby('FullName')['Points'].sum().to_dict()
    
    # 2b. Get each driver's 2025 performance (mean and std dev of points per race)
    # This will be our "statistical model" for each driver
    driver_stats = df_2025.groupby('FullName')['Points'].agg(['mean', 'std']).fillna(0)
    # If std is 0 (e.g., only 1 race), give it a default value to allow for variation
    driver_stats.loc[driver_stats['std'] == 0, 'std'] = driver_stats['mean'] / 2 
    driver_stats = driver_stats.to_dict('index')

    print("Current Top 5 Standings:")
    top_5 = sorted(current_standings.items(), key=lambda item: item[1], reverse=True)
    for driver, points in top_5[:5]:
        print(f"  {driver}: {points} points")

    # --- 3. GET REMAINING RACES ---
    print("\nFetching remaining 2025 F1 schedule...")
    try:
        schedule = ff1.get_event_schedule(CURRENT_YEAR)
        
        # Get today's date (with timezone) to compare
        today = pd.Timestamp.now(tz='UTC')
        
        if schedule['EventDate'].dt.tz is None:
            schedule['EventDate'] = schedule['EventDate'].dt.tz_localize('UTC')

        # Find all future races
        future_races = schedule[schedule['EventDate'] > today]
        races_left_count = len(future_races)
        
        # Check for Sprint races left (they award extra points)
        sprints_left_count = len(future_races[future_races['EventFormat'] == 'sprint'])
        
        print(f"Found {races_left_count} event(s) left:")
        for index, race in future_races.iterrows():
            print(f"  - {race['EventName']} ({race['EventFormat']})")
        
    except Exception as e:
        print(f"Could not load 2025 schedule from FastF1: {e}")
        print("Please ensure you are connected to the internet.")
        return

    if races_left_count == 0:
        print("Season is already over! No races to predict.")
        return

    # --- 4. RUN MONTE CARLO SIMULATION ---
    print(f"\n--- Running {SIMULATION_COUNT} Championship Simulations ---")
    
    # Store the final points tally for each driver in each simulation
    simulation_results = {name: [] for name in CONTENDERS}
    # Count how many times each contender wins the championship
    championship_wins = {name: 0 for name in CONTENDERS}
    
    all_drivers = list(driver_stats.keys())

    for i in range(SIMULATION_COUNT):
        # Start each simulation with the current points
        final_points = current_standings.copy()
        
        # Simulate each remaining race
        for index, race in future_races.iterrows():
            # In each race, every driver scores some points
            for driver_name in all_drivers:
                if driver_name not in driver_stats:
                    continue # Skip drivers with no 2025 data
                
                stats = driver_stats[driver_name]
                avg_points = stats['mean']
                std_dev_points = stats['std']
                
                # Get a simulated score from a normal distribution
                # This is the "model": it assumes drivers will perform around their 2025 average
                sim_points = np.random.normal(avg_points, std_dev_points)
                
                # Clamp the points (can't score less than 0 or more than 34)
                # 26 (P1 + FL) + 8 (Sprint P1) = 34
                sim_points = np.clip(sim_points, 0, 34)
                
                # Add the simulated points to their tally
                if driver_name in final_points:
                    final_points[driver_name] += sim_points
                else:
                    final_points[driver_name] = sim_points

        # --- Tally the winner of this one simulation ---
        # Find the driver with the most points
        winner_name = max(final_points, key=final_points.get)
        
        if winner_name in championship_wins:
            championship_wins[winner_name] += 1
            
        # Store the final scores for our contenders
        for name in CONTENDERS:
            simulation_results[name].append(final_points.get(name, 0))

    # --- 5. DISPLAY FINAL RESULTS ---
    print("\n--- Championship Simulation Complete ---")

    print("\n--- Predicted Final Points (Average) ---")
    for name in CONTENDERS:
        avg_final_points = np.mean(simulation_results[name])
        print(f"  {name}: {avg_final_points:.0f} points")

    print("\n--- Championship Win Probability ---")
    # Sort by win percentage
    sorted_winners = sorted(championship_wins.items(), key=lambda item: item[1], reverse=True)
    
    # Show the top contenders, even if they weren't in our initial list
    for name, wins in sorted_winners[:5]:
        if wins > 0:
            win_pct = (wins / SIMULATION_COUNT) * 100
            print(f"  {name}: {win_pct:.1f}%")

    # Answer your specific question
    ver_pct = (championship_wins.get('Max Verstappen', 0) / SIMULATION_COUNT) * 100
    print(f"\n> Max Verstappen's chance of winning: {ver_pct:.1f}%")

# Run the simulation
if __name__ == "__main__":
    run_championship_simulation()