from flask import Flask, render_template, url_for
import fastf1
import pandas as pd
from datetime import datetime, timedelta
import random

app = Flask(__name__)
fastf1.Cache.enable_cache('cache')

# --- CONFIGURATION ---
# Simulate Date: Wednesday before Brazil 2025
SIMULATED_NOW = datetime(2025, 11, 5)

# Mock Data: Driver Ratings (Higher = Better Odds)
DRIVER_RATINGS = {
    "Verstappen": 95, "Norris": 94, "Hamilton": 90, "Leclerc": 89,
    "Piastri": 88, "Russell": 85, "Antonelli": 75, "Alonso": 82,
    "Sainz": 70, "Bearman": 65, "Colapinto": 60
}

def get_race_data():
    schedule = fastf1.get_event_schedule(2025, include_testing=False)
    schedule['EventDate'] = pd.to_datetime(schedule['EventDate'])
    
    # Split races
    past = schedule[schedule['EventDate'] < SIMULATED_NOW]
    future = schedule[schedule['EventDate'] >= SIMULATED_NOW]
    
    # FIX: Convert .iloc[0] to a dictionary immediately
    if not future.empty:
        next_race = future.iloc[0].to_dict()
    else:
        next_race = None
    
    return past.to_dict('records'), future.to_dict('records'), next_race

@app.route('/')
def home():
    past, future, next_race = get_race_data()
    
    # Calculate days until next race for the countdown
    days_until = (next_race['EventDate'] - SIMULATED_NOW).days if next_race is not None else 0
    
    return render_template('index.html', 
                           past_races=past, 
                           future_races=future, 
                           next_race=next_race,
                           days_until=days_until,
                           now=SIMULATED_NOW)

@app.route('/standings')
def standings():
    # Simulated Championship Table
    standings_data = [
        {"pos": 1, "driver": "Lando Norris", "team": "McLaren", "points": 345, "wins": 6},
        {"pos": 2, "driver": "Max Verstappen", "team": "Red Bull", "points": 338, "wins": 7},
        {"pos": 3, "driver": "Charles Leclerc", "team": "Ferrari", "points": 290, "wins": 3},
        {"pos": 4, "driver": "Oscar Piastri", "team": "McLaren", "points": 265, "wins": 2},
        {"pos": 5, "driver": "Lewis Hamilton", "team": "Ferrari", "points": 210, "wins": 1},
        {"pos": 6, "driver": "George Russell", "team": "Mercedes", "points": 180, "wins": 1},
    ]
    return render_template('standings.html', standings=standings_data)

@app.route('/bets')
def bets():
    # Generate dynamic odds based on ratings
    drivers = []
    for name, rating in DRIVER_RATINGS.items():
        # Formula to convert rating to realistic decimal odds
        # Higher rating = Lower odds (Higher chance of winning)
        raw_odd = 100 / (rating - 40) 
        # Add some randomness
        final_odd = round(raw_odd + random.uniform(0, 0.5), 2)
        
        team_map = {"Verstappen": "Red Bull", "Norris": "McLaren", "Hamilton": "Ferrari", "Sainz": "Williams"}
        team = team_map.get(name, "Formula 1 Team")
        
        drivers.append({"name": name, "team": team, "odds_win": final_odd})

    # Sort by odds (favorites first)
    drivers.sort(key=lambda x: x['odds_win'])

    matchups = [
        {"p1": "Norris", "p2": "Verstappen", "odds1": 1.90, "odds2": 1.90},
        {"p1": "Hamilton", "p2": "Leclerc", "odds1": 2.10, "odds2": 1.65},
    ]

    return render_template('bets.html', drivers=drivers, matchups=matchups)

if __name__ == '__main__':
    app.run(debug=True)