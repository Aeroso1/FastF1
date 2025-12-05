import fastf1 as ff1
import fastf1.plotting
from fastf1.core import Laps

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import matplotlib
matplotlib.use("TkAgg")
#cache
cache_path = "cache"

if not os.path.exists(cache_path):
    os.makedirs(cache_path)
    print(f"Created cache directory at: {cache_path}")
ff1.Cache.enable_cache(cache_path)

#plotting style
fastf1.plotting.setup_mpl()

#load Brazil 2025
print("Loading session data...")
session = ff1.get_session(2025, "São Paulo", "R")
session.load()
print("Session data loaded")

#verstappen laps
all_laps = session.laps
ver_laps = all_laps.pick_drivers("VER")
#filter flying laps
ver_laps_clean = ver_laps.pick_quicklaps().copy()

#lap time in total seconds so we can plot it
ver_laps_clean["LapTimeSeconds"] = ver_laps_clean["LapTime"].dt.total_seconds()

#last stint
last_stint_num = ver_laps_clean["Stint"].max()
#controversial stint
controversial_stint = ver_laps_clean[ver_laps_clean["Stint"] == 3]
#puncture stint
puncture_stint = ver_laps[ver_laps["Stint"] == 1]

print(f"\n--- Stint Analysis (VER) ---")
print(f"Stint 1 (Puncture): {len(puncture_stint)} clean laps on {puncture_stint["Compound"].iloc[0]}")
print(f"Stint 3 (Controversial): {len(controversial_stint)} clean laps on {controversial_stint["Compound"].iloc[0]}")
try:
    stint_4_laps = ver_laps_clean[ver_laps_clean['Stint'] == 4]
    print(f"Stint 4 (Final Stint): {len(stint_4_laps)} clean laps on {stint_4_laps["Compound"].iloc[0]}")
except IndexError:
    print(f"Stint 4 (Final Stint): No 'clean' laps found (likely just pit laps). ")

###plot 2
print("\n--- Plotting Stint 1 (Puncture Analysis) ---")
puncture_stint_data = ver_laps[ver_laps["Stint"] == 1].copy()
puncture_stint_data["LapTimeSeconds"] = puncture_stint_data["LapTime"].dt.total_seconds()
### new plot figure
plt.figure(figsize=(10, 6))
plt.plot(puncture_stint_data["LapNumber"],
         puncture_stint_data["LapTimeSeconds"],
         "o-")
plt.title("Verstappen's Stint 1 - Puncture Analysis")
plt.xlabel("Lap Number")
plt.ylabel("Lap Time (Seconds)")
plt.grid(True)

plt.savefig("verstappen_puncture.png")
print("Puncture plot saved to verstappen_puncture.png")
#plt.show(block=True)
#input("Press ENTER in terminal to continue to next plot")

#plot tyre degg. y = LapTime (what we want to predict); x = TyreLife (what we know)
#"o" = dot; "--" = line

x_data = controversial_stint["TyreLife"].values.reshape(-1, 1)
y_data = controversial_stint["LapTimeSeconds"].values

print(f"Fitting model on {len(x_data)} data points...")

#create model (tries to find best straight line)
model = make_pipeline(PolynomialFeatures(degree=3), LinearRegression())
model.fit(x_data, y_data)

# VER pitted on lap 55. race was 71 laps
# 16 more laps to go
# tyre life on that last lap was ~20.
# we need to project from TyreLife 20 up to 36 (20 + 16).
# numpy array for all laps from 1 to 36.

#first plot
#min_tyre_life = int(x_data.min())
#max_tyre_life = int(x_data.max())
#laps_to_go_in_race = 16
#project_until = max_tyre_life + laps_to_go_in_race

#print(f"Data found from TyreLife {min_tyre_life} to {max_tyre_life}")
#print(f"Projecting {laps_to_go_in_race} more laps, up to TyreLife {project_until}...")
#laps_to_project = np.arange(min_tyre_life, project_until + 1).reshape(-1, 1)
###ask model to predict lap time for 36 laps
#predicted_laptimes = model.predict(laps_to_project)
#print("Projection complete.")

#plt.figure(figsize=(12, 7))
#plt.plot(x_data,
#         y_data,
#        "o",
#        color="blue",
#        label="Stint 3 - Actual Laps (VER)")

###plot_2 
#plt.plot(laps_to_project,
#        predicted_laptimes,
#        "r--",
#        label = "Projected Degradation")

#plt.title("Verstappen's Tyre Degradation and 'What if' Projection")
#plt.xlabel("Tyre Life (Laps)")
#plt.ylabel("Lap time (Seconds)")
#plt.legend()
#plt.grid(True)

#plt.savefig("verstappen_projection.png")
#print("\n--- Plot saved to verstappen_projection.png ---")
#plt.show(block=True)
#input("Press ENTER in your terminal to close")

#ghost race 
print("\n--- Simulating race Without Puncture ---")
#get "normal" pace (average of stint 2)
stint_2_laps = ver_laps_clean[ver_laps_clean["Stint"] == 2]
avg_pace = stint_2_laps["LapTimeSeconds"].mean()

#hypothetical race data
ver_race_laps = ver_laps.copy()
ver_race_laps["HypotheticalTime"] = ver_race_laps["LapTime"].dt.total_seconds()

#find any slow laps on Stint 1 and replace with "Normal Laps"
puncture_lap = 7
ver_race_laps.loc[ver_race_laps["LapNumber"] == puncture_lap, "HypotheticalTime"] = avg_pace
print(f"Replaced Lap {puncture_lap} with average pace ({avg_pace:.2f}s)")

#calculate total race time
ver_race_laps["ActualRaceTime"] = ver_race_laps["LapTime"].dt.total_seconds().cumsum()
ver_race_laps["HypotheticalRaceTime"] = ver_race_laps["HypotheticalTime"].cumsum()

#calculate the gap (time lost)
ver_race_laps["TimeLost"] = ver_race_laps["ActualRaceTime"] - ver_race_laps["HypotheticalRaceTime"]
final_time_saved = float(ver_race_laps["TimeLost"].iloc[-1])
print(f"Total Time Lost due to Puncture: {final_time_saved:.2f} seconds")

#plot 1 - ghost race 
plt.figure(figsize=(10, 6))

plt.plot(ver_race_laps["LapNumber"],
         ver_race_laps["TimeLost"],
         label="Gap to Ghost Car (Seconds Lost)",
         color = "red",
         linewidth=2)

plt.fill_between(ver_race_laps["LapNumber"],
                 ver_race_laps["TimeLost"],
                 color="red",
                 alpha=0.1)

plt.title(f"Cost of the Puncture: {final_time_saved:.1f} Seconds Lost")
plt.xlabel("Lap Number")
plt.ylabel("Seconds Behind Ghost Car")

plt.legend()
plt.grid(True)

plt.savefig("verstappen_ghost_race.png")
print("Ghost Race plot saved to verstappen_ghost_race.png")
plt.show(block=True)


