🏎️ Brazilian GP 2025 Analysis: The "What If" Project

A data-driven analysis of Max Verstappen's strategic decisions and "ghost race" scenarios.

Project Status: Complete

📖 Overview

The 2025 Brazilian Grand Prix was one of the most dramatic races of the season. Max Verstappen started from the pit lane, suffered an early puncture, and still climbed back to finish P3.

This project utilizes Python and FastF1 telemetry data to answer two critical strategic questions that left fans debating after the race:

The Strategy Call: Was pitting from the lead on Lap 55 a mistake, or were his tyres truly dead?

The "Ghost" Scenario: If he hadn't suffered that early puncture on Lap 7, would he have won the race?

🛠️ Tech Stack

This project relies on the modern Python data science stack:

FastF1: For accessing official F1 timing and telemetry data.

Pandas: For data manipulation, filtering laps, and time-series analysis.

Matplotlib: For visualizing lap times, tyre degradation curves, and gap analysis.

Scikit-learn: Used to build PolynomialFeatures regression models to predict tyre life cliffs.

📊 Analysis Part 1: The Tyre Cliff

Question: Was pitting from the lead on Lap 55 a mistake?

Verstappen gave up the lead on Lap 55 to pit for fresh tyres. To determine if this was necessary, we analyzed the degradation of his Medium tyres during Stint 3.

Methodology:

Isolated "clean" laps (removing Safety Car or traffic-impacted laps).

Plotted Lap Time vs. Tyre Age.

Trained a Machine Learning model (Polynomial Regression, degree=3) to project future lap times if he had stayed out.

Findings:
The data revealed a massive "Tyre Cliff." On the final lap of the stint (Tyre Life 20), his pace dropped by ~3.5 seconds compared to previous laps. Our ML model predicted that staying out would have resulted in lap times ballooning to 90s, 100s, and 120s+ within 15 laps.

(Note: Pitting was the only viable option. Staying out would have been catastrophic.)

👻 Analysis Part 2: The "Ghost Race" Simulation

Question: Could he have won without the puncture?

Max's race was compromised on Lap 7 due to a puncture that forced an unscheduled pit stop. We created a simulation to calculate exactly how much time this cost him.

Methodology:

Isolate the Event: Identified Lap 7 as the anomaly.

Calculate True Pace: Calculated Verstappen's average clean lap time from his recovery drive (Stint 2).

Data Imputation: We created a "Ghost Race" dataset where we replaced only the Lap 7 puncture time with his "True Pace" average. All other variables (traffic, safety cars, strategy) remained identical to reality.

Gap Analysis: We calculated the cumulative sum (.cumsum()) of the time difference between the Real Max and Ghost Max.

Findings:
The simulation shows a flat line gap of 0 seconds until Lap 7, where it instantly spikes to 32.5 seconds.

Why is the line flat?
This indicates a conservative model. After the puncture, "Ghost Max" matches "Real Max" lap-for-lap. This proves that the puncture alone—independent of any other race factors—created a permanent 32.5-second deficit.

🏆 Final Verdict

By combining the Official Race Results with our Simulation Data, we can definitively solve the "What If" scenario.

Metric

Time / Result

Official Gap to Winner (Lando Norris)

+10.750 s

Time Lost to Puncture (Calculated)

32.500 s

Net Result (Without Puncture)

-21.750 s

Conclusion

Our analysis proves that the puncture cost Max Verstappen 32.5 seconds. Since he finished only 10.75 seconds behind the winner, we can conclude with high confidence that Max Verstappen would have won the 2025 Brazilian Grand Prix by approximately 21 seconds if he had not suffered the puncture.
