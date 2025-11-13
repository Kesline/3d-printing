#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Basic NSGA-II for PLA 3D Printing (Kesline)

- Loads ../data/printer_data_log-2.csv
- Trains two RandomForest regressors:
    * job/time_printing  (minimize)
    * surface_quality    (maximize)
- Runs NSGA-II (DEAP) over decision variables:
    * printer/temp_nozzle
    * printer/temp_bed
    * printer/fan_print
  (printer/speed and printer/flow are fixed to 100)
- Saves:
    * ../week7_outputs/pareto_front_basic.csv
    * ../week7_outputs/pareto_front_basic.png
"""

import os
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from deap import base, creator, tools, algorithms

# ---------------------------------------------------------------------
# 1. Paths & data loading
# ---------------------------------------------------------------------

# Script is in:     3d-printer 2025 fall/model/
# CSV is in:        3d-printer 2025 fall/data/
# Outputs go in:    3d-printer 2025 fall/week7_outputs/
CSV_PATH = "../data/printer_data_log-2.csv"
OUTPUT_DIR = "../week7_outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Loading data from: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)

# Keep only rows where both objectives are defined
mask = df["surface_quality"].notna() & df["job/time_printing"].notna()
df = df.loc[mask].copy()

# Decision variables (we fix speed & flow at 100 later)
FEATURES = [
    "printer/temp_nozzle",
    "printer/temp_bed",
    "printer/fan_print",
    "printer/speed",
    "printer/flow",
]

TARGET_TIME = "job/time_printing"
TARGET_QUALITY = "surface_quality"

df = df[FEATURES + [TARGET_TIME, TARGET_QUALITY]].dropna()

print(f"After filtering/dropna: {len(df)} rows")

# Split into X and y
X = df[FEATURES]
y_time = df[TARGET_TIME]
y_quality = df[TARGET_QUALITY]

# Train/validation split (keep y_time and y_quality aligned)
X_train, X_val, y_time_train, y_time_val, y_quality_train, y_quality_val = train_test_split(
    X, y_time, y_quality, test_size=0.2, random_state=42
)

# ---------------------------------------------------------------------
# 2. Train Random Forest models for time and quality
# ---------------------------------------------------------------------

rf_time = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
)
rf_time.fit(X_train, y_time_train)

rf_quality = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
)
rf_quality.fit(X_train, y_quality_train)

# Quick sanity check
print("\nExample predictions on a validation point:")
print("  True time:    ", y_time_val.iloc[0])
print("  Pred time:    ", rf_time.predict(X_val.iloc[[0]])[0])
print("  True quality: ", y_quality_val.iloc[0])
print("  Pred quality: ", rf_quality.predict(X_val.iloc[[0]])[0])

# ---------------------------------------------------------------------
# 3. Define NSGA-II search space
# ---------------------------------------------------------------------
# We optimize ONLY: [temp_nozzle, temp_bed, fan_print]
# and set speed=100, flow=100 (constant in our model eval).

BOUNDS = {
    "printer/temp_nozzle": (200.0, 230.0),   # °C
    "printer/temp_bed": (55.0, 61.0),        # °C
    # fan_print range based on your logs (0–6067)
    "printer/fan_print": (0.0, 6067.0),
}

DECISION_KEYS = [
    "printer/temp_nozzle",
    "printer/temp_bed",
    "printer/fan_print",
]

BOUNDS_LOW = [BOUNDS[k][0] for k in DECISION_KEYS]
BOUNDS_HIGH = [BOUNDS[k][1] for k in DECISION_KEYS]
NDIM = len(DECISION_KEYS)

# ---------------------------------------------------------------------
# 4. DEAP setup: NSGA-II
# ---------------------------------------------------------------------

# Protect against re-running in same interpreter
if "FitnessMulti" not in creator.__dict__:
    creator.create("FitnessMulti", base.Fitness, weights=(-1.0, 1.0))  # min time, max quality
if "Individual" not in creator.__dict__:
    creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()

# Attribute generator: random value within bounds for each variable
for i in range(NDIM):
    low, up = BOUNDS_LOW[i], BOUNDS_HIGH[i]
    toolbox.register(f"attr_float_{i}", random.uniform, low, up)


def init_individual():
    """Create one individual with NDIM decision variables."""
    return creator.Individual([
        toolbox.__getattribute__(f"attr_float_{i}")()
        for i in range(NDIM)
    ])


toolbox.register("individual", init_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def evaluate(individual):
    """
    Evaluates an individual:
    - Build a row with nozzle, bed, fan, speed=100, flow=100
    - Predict time and quality with RF models
    - Return (time_pred, quality_pred)
    DEAP uses weights (-1, +1) so this becomes:
      minimize time, maximize quality
    """
    nozzle, bed, fan = individual

    row = pd.DataFrame([{
        "printer/temp_nozzle": nozzle,
        "printer/temp_bed": bed,
        "printer/fan_print": fan,
        "printer/speed": 100.0,
        "printer/flow": 100.0,
    }])

    time_pred = float(rf_time.predict(row)[0])
    quality_pred = float(rf_quality.predict(row)[0])

    return time_pred, quality_pred


toolbox.register("evaluate", evaluate)

# NSGA-II operators
toolbox.register(
    "mate",
    tools.cxSimulatedBinaryBounded,
    low=BOUNDS_LOW,
    up=BOUNDS_HIGH,
    eta=20.0,
)
toolbox.register(
    "mutate",
    tools.mutPolynomialBounded,
    low=BOUNDS_LOW,
    up=BOUNDS_HIGH,
    eta=20.0,
    indpb=1.0 / NDIM,
)
toolbox.register("select", tools.selNSGA2)

# ---------------------------------------------------------------------
# 5. Run NSGA-II
# ---------------------------------------------------------------------


def main():
    random.seed(42)

    POP_SIZE = 80
    NGEN = 40
    CXPB = 0.9
    MUTPB = 0.1

    print("\nStarting NSGA-II optimization...")
    pop = toolbox.population(n=POP_SIZE)
    # NSGA-II needs an initial selection to assign crowding distance
    pop = toolbox.select(pop, len(pop))

    # Keep Pareto front
    hof = tools.ParetoFront()

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    pop, logbook = algorithms.eaMuPlusLambda(
        population=pop,
        toolbox=toolbox,
        mu=POP_SIZE,
        lambda_=POP_SIZE,
        cxpb=CXPB,
        mutpb=MUTPB,
        ngen=NGEN,
        stats=stats,
        halloffame=hof,
        verbose=True,
    )

    # -----------------------------------------------------------------
    # 6. Save Pareto front to CSV
    # -----------------------------------------------------------------
    pareto_rows = []
    for ind in hof:
        nozzle, bed, fan = ind
        time_pred, quality_pred = ind.fitness.values
        pareto_rows.append({
            "printer/temp_nozzle": nozzle,
            "printer/temp_bed": bed,
            "printer/fan_print": fan,
            "pred_time": time_pred,
            "pred_surface_quality": quality_pred,
        })

    pareto_df = pd.DataFrame(pareto_rows)
    pareto_df.sort_values(by="pred_time", inplace=True)

    csv_path = os.path.join(OUTPUT_DIR, "pareto_front_basic.csv")
    pareto_df.to_csv(csv_path, index=False)
    print(f"\nSaved Pareto front CSV to: {csv_path}")
    print(pareto_df.head())

    # -----------------------------------------------------------------
    # 7. Plot Pareto front and save PNG
    # -----------------------------------------------------------------
    plt.figure()
    plt.scatter(
        pareto_df["pred_time"],
        pareto_df["pred_surface_quality"],
        s=20,
        alpha=0.7,
    )
    plt.xlabel("Predicted Print Time (job/time_printing)")
    plt.ylabel("Predicted Surface Quality")
    plt.title("NSGA-II Pareto Front: Time vs Surface Quality")
    plt.grid(True)

    png_path = os.path.join(OUTPUT_DIR, "pareto_front_basic.png")
    plt.tight_layout()
    plt.savefig(png_path, dpi=300)
    plt.close()
    print(f"Saved Pareto front plot to: {png_path}")

    return pop, hof, logbook


if __name__ == "__main__":
    main()
