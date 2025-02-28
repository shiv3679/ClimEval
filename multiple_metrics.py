#!/usr/bin/env python3

import os
import sys
import math
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import cartopy.feature as cfeature

# For distribution metrics
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon

###########################################################
#                METRIC CALCULATION FUNCTIONS
###########################################################

########################
# 1. Continuous Metrics
########################
def compute_continuous_metrics(model_data, obs_data):
    """
    Computes standard continuous metrics:
     - Mean Squared Error (MSE)
     - Mean Absolute Error (MAE)
     - Bias (mean error)
     - Pearson Correlation Coefficient (Corr)
       (If you want 'ACC' specifically, you'd compare anomalies from climatology.)
    Returns a dictionary of results.
    """
    # Flatten to 1D for simplicity
    m_arr = model_data.values.flatten()
    o_arr = obs_data.values.flatten()

    # MSE
    mse = np.mean((m_arr - o_arr) ** 2)
    # MAE
    mae = np.mean(np.abs(m_arr - o_arr))
    # Bias
    bias = np.mean(m_arr - o_arr)
    # Correlation
    corr_matrix = np.corrcoef(m_arr, o_arr)
    corr = corr_matrix[0, 1]

    return {
        'MSE': mse,
        'MAE': mae,
        'Bias': bias,
        'Corr': corr
    }


########################
# 2. Event-Based Metrics
########################
def compute_event_metrics(model_data, obs_data, threshold=1.0, neighborhood_size=None):
    """
    Computes common event-based metrics for binary phenomena (yes/no events).
    We define an event as model_data >= threshold, similarly for obs_data.
    
     - POD (Probability of Detection)
     - FAR (False Alarm Ratio)
     - CSI (Critical Success Index) / ETS (Equitable Threat Score)
     - Simplified Fractions Skill Score (FSS)
       * This is a minimal example. Real FSS often uses sliding windows or multiple scales.
       
    We assume the data is the same shape and time dimension, but we flatten for overall statistics.
    threshold: the cutoff above which we call it an "event" (e.g., precip > 1 mm)
    neighborhood_size: if not None, we do a very simple "box" FSS approach.
    
    Returns a dictionary with these metrics.
    """
    # Flatten
    m_arr = model_data.values.flatten()
    o_arr = obs_data.values.flatten()

    # Convert to binary events
    model_event = (m_arr >= threshold).astype(int)
    obs_event   = (o_arr >= threshold).astype(int)

    # Basic contingency table elements
    hits         = np.sum((model_event == 1) & (obs_event == 1))
    misses       = np.sum((model_event == 0) & (obs_event == 1))
    false_alarms = np.sum((model_event == 1) & (obs_event == 0))
    # correct_negatives = np.sum((model_event == 0) & (obs_event == 0)) # sometimes needed

    # POD
    if (hits + misses) == 0:
        pod = np.nan
    else:
        pod = hits / (hits + misses)

    # FAR
    if (hits + false_alarms) == 0:
        far = np.nan
    else:
        far = false_alarms / (hits + false_alarms)

    # CSI (same as TS)
    denom = hits + misses + false_alarms
    if denom == 0:
        csi = np.nan
    else:
        csi = hits / denom

    # ETS (Equitable Threat Score)
    # ETS = (hits - hits_random) / (hits + misses + false_alarms - hits_random)
    # hits_random = (hits+misses)*(hits+false_alarms)/(hits+misses+false_alarms+correct_negatives)
    # For a large domain, you can approximate total grid points = hits+misses+false_alarms+correct_negatives
    total = len(m_arr)
    correct_negatives = total - (hits + misses + false_alarms)
    hits_random = ((hits + misses) * (hits + false_alarms)) / total if total else 0
    ets_denom   = hits + misses + false_alarms - hits_random
    if ets_denom == 0:
        ets = np.nan
    else:
        ets = (hits - hits_random) / ets_denom

    # Simplified Fractions Skill Score (FSS) example:
    # Real FSS usually requires a neighborhood approach for each grid point. 
    # Here we do a placeholder that just compares the fraction of event points in each dataset.
    # If you want a 2D neighborhood-based approach, you'd need to do a sliding window or similar.
    frac_model = np.mean(model_event)
    frac_obs   = np.mean(obs_event)
    fss = 1.0 - np.mean((model_event - obs_event)**2) / (np.mean(model_event**2) + np.mean(obs_event**2))

    # If we do a more "neighborhood" approach, we must define a method to average model_event, obs_event
    # in sliding windows, then compare. That is more advanced. This is a minimal placeholder.

    return {
        'POD': pod,
        'FAR': far,
        'CSI': csi,
        'ETS': ets,
        'FSS': fss
    }


########################
# 3. Probabilistic Metrics
########################
def compute_probabilistic_metrics(model_prob, obs_event, reference_prob=None):
    """
    Suppose 'model_prob' is an array of predicted probabilities (0..1),
    and 'obs_event' is 0/1 for whether the event occurred.
    
     - Brier Score (BS) = mean( (p - o)^2 )
     - Brier Skill Score (BSS) = 1 - BS/BS_ref
       (where BS_ref might be the climatological probability, or user-provided reference)
     - Ranked Probability Skill Score (RPSS): For multi-category or multi-threshold
       forecasts, we do a simplified version here for demonstration.
    
    This is a demonstration function. In real usage, you'd need
    the forecast distribution for multiple categories (e.g., terciles)
    to properly compute RPSS.
    """
    # Flatten
    p_arr = model_prob.values.flatten()
    o_arr = obs_event.values.flatten().astype(float)

    # Brier Score
    bs = np.mean((p_arr - o_arr)**2)

    # If no reference prob is given, we can guess it from the mean observed frequency
    if reference_prob is None:
        reference_prob = np.mean(o_arr)

    # Brier Score for reference
    bs_ref = np.mean((reference_prob - o_arr)**2)
    if bs_ref == 0:
        bss = np.nan
    else:
        bss = 1.0 - (bs / bs_ref)

    # For an actual RPSS, we'd need multiple categories and cumulative probabilities.
    # We'll show a simple placeholder that returns np.nan for now.
    rpss = np.nan  # placeholder

    return {
        'BS': bs,
        'BSS': bss,
        'RPSS': rpss
    }


########################
# 4. Distribution-Level
########################
def compute_distribution_metrics(model_data, obs_data, bins=50):
    """
    Compares distributions of model vs. obs for extremes or overall shape:
     - Wasserstein distance (a.k.a. Earth Mover's Distance)
     - Jensen-Shannon divergence (JS)
    
    We flatten the arrays, then compute empirical distributions. 
    Alternatively, we can do a direct sample-based approach for Wasserstein:
      W = wasserstein_distance(m_arr, o_arr)
    For JS, we need discrete probability distributions (histogram-based).
    """
    m_arr = model_data.values.flatten()
    o_arr = obs_data.values.flatten()

    # 1) Wasserstein distance: SciPy can do sample-based directly.
    wdist = wasserstein_distance(m_arr, o_arr)

    # 2) Jensen-Shannon Divergence: needs probability distributions
    # We'll do a quick histogram approach, then convert to probabilities.
    min_val = min(np.min(m_arr), np.min(o_arr))
    max_val = max(np.max(m_arr), np.max(o_arr))
    # Create histograms
    m_hist, edges = np.histogram(m_arr, bins=bins, range=(min_val, max_val), density=True)
    o_hist, _     = np.histogram(o_arr, bins=bins, range=(min_val, max_val), density=True)

    # jensenshannon returns the sqrt(JS divergence) by default
    js_dist = jensenshannon(m_hist, o_hist)

    return {
        'Wasserstein': wdist,
        'JensenShannon': js_dist
    }


###########################################################
#                 SUPPORTING FUNCTIONS
###########################################################

def validate_variable_existence(ds, var_name):
    if var_name not in ds.data_vars:
        raise KeyError(f"Variable '{var_name}' not found in dataset. "
                       f"Available: {list(ds.data_vars.keys())}")


def check_dataset_dims(model_ds, obs_ds, var_name):
    """
    Simple check for lat, lon, time dimension existence & shape.
    If mismatch is found, we just print a warning or raise an error.
    You could add regridding or interpolation logic here if you want.
    """
    required_dims = ['time', 'lat', 'lon']
    for dim in required_dims:
        if dim not in model_ds.dims:
            raise ValueError(f"Model dataset missing dimension: {dim}")
        if dim not in obs_ds.dims:
            raise ValueError(f"Obs dataset missing dimension: {dim}")

    # Check shapes quickly
    for dim in required_dims:
        if model_ds.sizes[dim] != obs_ds.sizes[dim]:
            print(f"Warning: dimension size mismatch for {dim}. "
                  "Consider regridding or manual fix.")
            # We'll not forcibly regrid in this script, just warn.


def visualize_map(model_data, obs_data, time_index=0):
    """
    Minimal example of a map-based visualization using Cartopy
    to see model vs. obs for a single time step.
    """
    proj = ccrs.PlateCarree()

    fig = plt.figure(figsize=(16, 5))

    # Subplot 1: Model
    ax1 = fig.add_subplot(1, 2, 1, projection=proj)
    ax1.set_title(f"Model (time={time_index})")
    ax1.coastlines()
    ax1.add_feature(cfeature.BORDERS, linestyle=':')
    m_slice = model_data.isel(time=time_index)
    m_slice.plot(
        ax=ax1, transform=proj,
        x='lon', y='lat', cmap='viridis',
        cbar_kwargs={"label": model_data.name}
    )

    # Subplot 2: Obs
    ax2 = fig.add_subplot(1, 2, 2, projection=proj)
    ax2.set_title(f"Observations (time={time_index})")
    ax2.coastlines()
    ax2.add_feature(cfeature.BORDERS, linestyle=':')
    o_slice = obs_data.isel(time=time_index)
    o_slice.plot(
        ax=ax2, transform=proj,
        x='lon', y='lat', cmap='viridis',
        cbar_kwargs={"label": obs_data.name}
    )

    plt.tight_layout()
    plt.show()


###########################################################
#                     MAIN SCRIPT
###########################################################

def main():
    print("==== Climate Model vs. Observation Verification ====\n")

    # 1) Get user inputs
    model_path = input("Path to MODEL dataset: ").strip()
    obs_path   = input("Path to OBS dataset: ").strip()
    
    if not os.path.isfile(model_path):
        sys.exit(f"Model file not found: {model_path}")
    if not os.path.isfile(obs_path):
        sys.exit(f"Obs file not found: {obs_path}")

    # 2) Open datasets
    model_ds = xr.open_dataset(model_path)
    obs_ds   = xr.open_dataset(obs_path)

    # 3) Pick a variable to compare
    model_vars = list(model_ds.data_vars.keys())
    if not model_vars:
        sys.exit("No variables in model dataset.")
    
    print("\nAvailable model variables:")
    for idx, var in enumerate(model_vars):
        print(f"  [{idx}] {var}")
    var_choice = input("Enter index or name of variable to compare: ").strip()

    if var_choice.isdigit():
        var_idx = int(var_choice)
        if var_idx < 0 or var_idx >= len(model_vars):
            sys.exit("Invalid index for variable.")
        variable_name = model_vars[var_idx]
    else:
        if var_choice not in model_ds.data_vars:
            sys.exit(f"Variable {var_choice} not found in model dataset.")
        variable_name = var_choice

    # Check existence in obs
    if variable_name not in obs_ds.data_vars:
        sys.exit(f"Variable '{variable_name}' not found in obs dataset: {list(obs_ds.data_vars.keys())}")

    print(f"\nSelected variable: {variable_name}")

    # 4) Check dimension shapes or regrid if needed
    try:
        check_dataset_dims(model_ds, obs_ds, variable_name)
    except ValueError as e:
        print(f"Dimension check error: {e}")
        print("Proceeding, but be aware of mismatch.\n")

    # 5) Extract the DataArrays
    model_da = model_ds[variable_name]
    obs_da   = obs_ds[variable_name]

    # 6) Visualization for a chosen time step
    time_idx_str = input("\nEnter a time index to visualize [default=0]: ").strip()
    if not time_idx_str:
        time_idx = 0
    else:
        time_idx = int(time_idx_str)
    try:
        visualize_map(model_da, obs_da, time_index=time_idx)
    except Exception as e:
        print(f"Visualization error: {e}")

    # 7) Ask user which metric categories to compute
    print("\nWhich metrics do you want to compute? (You can select multiple)")
    print("1) Continuous (MSE, MAE, Bias, Corr)")
    print("2) Event-based (POD, FAR, CSI, ETS, FSS)")
    print("3) Probabilistic (Brier Score, BSS, RPSS)")
    print("4) Distribution-level (Wasserstein, Jensen-Shannon)")
    selection = input("Enter numbers separated by commas (e.g., '1,2'): ").strip()
    metric_choices = [x.strip() for x in selection.split(',')]

    # 7.1) Continuous metrics
    if '1' in metric_choices:
        cont_res = compute_continuous_metrics(model_da, obs_da)
        print("\n--- Continuous Metrics ---")
        for k, v in cont_res.items():
            print(f"{k}: {v:.4f}")

    # 7.2) Event-based
    if '2' in metric_choices:
        # We need a threshold to define events
        thr_str = input("Enter threshold for event-based metrics (default=1.0): ").strip()
        if not thr_str:
            thr_val = 1.0
        else:
            thr_val = float(thr_str)

        evt_res = compute_event_metrics(model_da, obs_da, threshold=thr_val)
        print("\n--- Event-Based Metrics ---")
        for k, v in evt_res.items():
            if isinstance(v, float) and not np.isnan(v):
                print(f"{k}: {v:.4f}")
            else:
                print(f"{k}: {v}")

    # 7.3) Probabilistic
    if '3' in metric_choices:
        # We assume model_da is a probability field (0..1) and obs_da is 0/1 events
        # or we ask the user if they'd like to interpret them that way
        # (This is an advanced scenario. Typically, you'd have an ensemble or separate prob variable.)
        prob_warning = input("\nIs your model data actually probabilities (y/n)? [default=n]: ").strip().lower()
        if prob_warning == 'y':
            # interpret model_da as probabilities, obs_da as 0/1
            # We might need to threshold obs to 0/1 as well
            obs_threshold = input("Enter threshold to convert obs to 0/1 [default=0.5]? ").strip()
            if not obs_threshold:
                obs_threshold_val = 0.5
            else:
                obs_threshold_val = float(obs_threshold)
            obs_event = (obs_da >= obs_threshold_val).astype(int)
            # no reference prob means we'll derive from obs
            prob_res = compute_probabilistic_metrics(model_da, obs_event)
            print("\n--- Probabilistic Metrics ---")
            for k, v in prob_res.items():
                if isinstance(v, float) and not np.isnan(v):
                    print(f"{k}: {v:.4f}")
                else:
                    print(f"{k}: {v}")
        else:
            print("Skipping probabilistic metrics since data is not probabilities.\n")

    # 7.4) Distribution-level
    if '4' in metric_choices:
        dist_res = compute_distribution_metrics(model_da, obs_da)
        print("\n--- Distribution Metrics ---")
        for k, v in dist_res.items():
            print(f"{k}: {v:.4f}")

    print("\nDone. Exiting normally.")


if __name__ == "__main__":
    main()
