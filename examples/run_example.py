"""
ClimEval - Climate Model Evaluation Example
===========================================

📌 **What this script does:**
1️⃣ Loads model & observational NetCDF datasets
2️⃣ Checks & fixes mismatches (dimensions, coordinates)
3️⃣ Computes climate verification metrics:
    - Continuous: MAE, MSE, Bias, Correlation
    - Event-based: POD, FAR, CSI, ETS
4️⃣ Generates **visualizations** (model vs. obs, bias maps)
5️⃣ Demonstrates **Dask parallel processing** (optional)

🔹 **Run from CLI**: `python examples/run_example.py`
"""

import os
import numpy as np
import xarray as xr
import dask.array as da
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Import ClimEval package
import climeval
from climeval.utils import open_dataset_with_chunks, detect_or_fix_mismatch
from climeval.core import compute_continuous_metrics, compute_event_metrics
from climeval.visualization import visualize_map, visualize_metric_map
from climeval.dask_setup import init_dask_cluster

# ✅ Dataset Paths (Modify as needed)
MODEL_PATH = "../../Datasets/india_model_output_005deg_irregular_storm.nc"
OBS_PATH = "../../Datasets/india_obs_output_005deg_irregular_storm.nc"

# ✅ Set Variable Name (Ensure it exists in your dataset)
VARIABLE_NAME = "lightning_density"

# ✅ Use Dask? (Toggle to True for parallel processing)
USE_DASK = True
USE_GPU = False  # Set True if using GPUs

def main():
    """Main function to run the example."""

    ##############################################################################
    # 1️⃣ INITIALIZE DASK CLUSTER (OPTIONAL)
    ##############################################################################

    if USE_DASK:
        print("\n✅ Step 1: Initializing Dask cluster...")
        client = init_dask_cluster(use_gpu=USE_GPU, n_workers=2)
        print(f"Dask dashboard available at: {client.dashboard_link}\n")

    ##############################################################################
    # 2️⃣ LOAD DATASETS (WITH OR WITHOUT DASK)
    ##############################################################################

    print("\n✅ Step 2: Loading datasets...")
    model_ds = open_dataset_with_chunks(MODEL_PATH, chunks={"time": 10, "lat": 200, "lon": 200})
    obs_ds = open_dataset_with_chunks(OBS_PATH, chunks={"time": 10, "lat": 200, "lon": 200})
    print("✅ Datasets loaded successfully!")

    ##############################################################################
    # 3️⃣ CHECK & FIX MISMATCHES (REGRID IF NEEDED)
    ##############################################################################

    print("\n✅ Step 3: Checking for mismatches...")
    model_ds, obs_ds = detect_or_fix_mismatch(model_ds, obs_ds)
    print("✅ Mismatch check completed!")

    ##############################################################################
    # 4️⃣ COMPUTE METRICS
    ##############################################################################

    model_da = model_ds[VARIABLE_NAME]
    obs_da = obs_ds[VARIABLE_NAME]

    # Continuous Metrics
    print("\n✅ Step 4: Computing continuous metrics...")
    cont_metrics = compute_continuous_metrics(model_da, obs_da)
    print("📊 Continuous Metrics:", cont_metrics)

    # Event-Based Metrics
    print("\n✅ Step 5: Computing event-based metrics...")
    event_metrics = compute_event_metrics(model_da, obs_da, threshold=1.0)
    print("📊 Event-Based Metrics:", event_metrics)

    ##############################################################################
    # 5️⃣ VISUALIZATIONS
    ##############################################################################

    # Side-by-side comparison of model vs. observations
    print("\n✅ Step 6: Displaying sample visualizations...")
    visualize_map(model_da, obs_da, time_index=0)

    # Bias Map Visualization
    print("\n✅ Step 7: Displaying Bias Map...")
    bias_map = (model_da - obs_da).mean(dim="time")
    bias_map = bias_map.compute() if USE_DASK else bias_map
    visualize_metric_map(bias_map, title_str="Bias Map", cmap="RdBu", center=0)

    ##############################################################################
    # 6️⃣ CLEANUP DASK (IF USED)
    ##############################################################################

    if USE_DASK:
        print("\n✅ Closing Dask client...")
        client.close()
        print("✅ Dask client closed successfully!")

    print("\n🎉 **ClimEval Example Run Complete!** 🚀")


# ✅ MAIN GUARD to Prevent Multiprocessing Errors
if __name__ == "__main__":
    main()
