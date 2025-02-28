#!/usr/bin/env python3

import os
import sys
import math
import numpy as np
import xarray as xr
import dask.array as da
import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon

# Dask distributed for parallelism
from dask.distributed import Client, LocalCluster

# Attempt to import CuPy for GPU usage
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

##############################################################################
#                        DASK & GPU SETUP
##############################################################################

def init_dask_cluster(use_gpu: bool = False, n_workers: int = 1):
    """
    Initializes a local Dask cluster (either CPU or GPU).
    Returns a dask.distributed.Client object.
    """
    if use_gpu:
        print("Initializing Dask LocalCluster for GPU usage...")
        # If you have multiple GPUs, you can set n_workers accordingly
        cluster = LocalCluster(
            n_workers=n_workers,
            threads_per_worker=1,  # typically 1 thread per GPU device
            processes=False
        )
    else:
        print("Initializing Dask LocalCluster for CPU usage...")
        cluster = LocalCluster(
            n_workers=n_workers,
            threads_per_worker=os.cpu_count() // n_workers if n_workers > 0 else 1
        )
    client = Client(cluster)
    print(f"Dask dashboard link: {client.dashboard_link}\n")
    return client


def open_dataset_with_chunks(path, chunks=None, use_gpu=False):
    """
    Opens a NetCDF file with xarray, applying Dask chunking.
    If use_gpu=True and CuPy is available, convert the dask array blocks to CuPy.
    """
    if chunks is None:
        # Some default chunk sizes (tweak as needed)
        chunks = {'time': 10, 'lat': 200, 'lon': 200}

    print(f"Opening dataset at {path} with chunks={chunks}")
    ds = xr.open_dataset(path, chunks=chunks)

    if use_gpu and GPU_AVAILABLE:
        print("Attempting to convert dataset to GPU arrays (CuPy)...")
        for var in ds.data_vars:
            ds[var].data = ds[var].data.map_blocks(cp.asarray, dtype=ds[var].dtype)
    else:
        if use_gpu and not GPU_AVAILABLE:
            print("Requested GPU usage, but CuPy not available. Continuing on CPU.")

    return ds

##############################################################################
#                REGRID & DIMENSION CHECK FUNCTIONS
##############################################################################

def detect_or_fix_mismatch(model_ds, obs_ds, required_dims=('time','lat','lon')):
    """
    Checks dimension mismatch for time, lat, lon.
    If mismatch is found, user can choose to regrid one dataset onto the other or exit.
    Returns updated model_ds, obs_ds that match in shape/coords (if regridding is done).
    """
    for dim in required_dims:
        if dim not in model_ds.dims:
            raise ValueError(f"Model dataset missing dimension: {dim}")
        if dim not in obs_ds.dims:
            raise ValueError(f"Observation dataset missing dimension: {dim}")

    # Check if shapes match
    same_shape = True
    for dim in required_dims:
        if model_ds.sizes[dim] != obs_ds.sizes[dim]:
            same_shape = False
            break

    if same_shape:
        # Possibly check coordinate arrays too. We'll load them in memory (careful with large data).
        print("Datasets have same dimension sizes. Checking coordinate values (may require compute).")
        # For large data, might do partial checks or handle lazily.
        # We'll do a naive approach that loads the entire coordinate array:
        model_lat = model_ds['lat'].values
        model_lon = model_ds['lon'].values
        model_time= model_ds['time'].values

        obs_lat = obs_ds['lat'].values
        obs_lon = obs_ds['lon'].values
        obs_time= obs_ds['time'].values

        coords_match = True
        if not np.allclose(model_lat, obs_lat):
            coords_match = False
        if not np.allclose(model_lon, obs_lon):
            coords_match = False
        if not np.array_equal(model_time, obs_time):
            coords_match = False

        if coords_match:
            print("Dimension sizes and coordinate arrays match. No regridding needed.\n")
            return model_ds, obs_ds
        else:
            print("Dimension sizes match, but coordinate values differ.\n")
    else:
        print("Dimension sizes differ.\n")

    # If we get here, there's a mismatch in size or coords
    choice = input("Mismatch detected. Regrid? (1) Regrid or (2) Exit [default=2]: ").strip()
    if choice == '1':
        direction = input("Regrid 'model->obs' (m2o) or 'obs->model' (o2m)? [default=m2o]: ").strip().lower()
        if direction == 'o2m':
            print("Regridding Observations -> Model coords...")
            obs_ds = obs_ds.interp_like(model_ds, method='linear')
        else:
            print("Regridding Model -> Observation coords...")
            model_ds = model_ds.interp_like(obs_ds, method='linear')
        print("Regridding complete.\n")
        return model_ds, obs_ds
    else:
        sys.exit("Exiting. Please reformat or align your data manually.")


##############################################################################
#                   METRIC CALCULATION FUNCTIONS
##############################################################################

########################
# 1. Continuous Metrics
########################
def compute_continuous_metrics(model_data, obs_data):
    """
    - Mean Squared Error (MSE)
    - Mean Absolute Error (MAE)
    - Bias
    - Pearson Correlation Coefficient
    For large Dask arrays, we .compute() at the end.
    """
    # Convert to dask arrays if not already
    m_arr = model_data.data.reshape(-1)
    o_arr = obs_data.data.reshape(-1)

    mse_dask = ((m_arr - o_arr)**2).mean()
    mae_dask = (da.fabs(m_arr - o_arr)).mean()
    bias_dask= (m_arr - o_arr).mean()

    # correlation
    # Using a standard approach with covariance
    mean_m = m_arr.mean()
    mean_o = o_arr.mean()
    cov = ((m_arr - mean_m)*(o_arr - mean_o)).mean()
    std_m = da.sqrt(((m_arr - mean_m)**2).mean())
    std_o = da.sqrt(((o_arr - mean_o)**2).mean())
    corr_dask = cov/(std_m*std_o)

    mse, mae, bias, corr = da.compute(mse_dask, mae_dask, bias_dask, corr_dask)
    return {
        'MSE': mse,
        'MAE': mae,
        'Bias': bias,
        'Corr': corr
    }

########################
# 2. Event-Based Metrics
########################
def compute_event_metrics(model_data, obs_data, threshold=1.0):
    """
    For binary events. We define an event if data >= threshold.
    Metrics: POD, FAR, CSI, ETS, FSS (placeholder).
    We'll load them into memory at the end, so watch out for large data.
    """
    m_flat = model_data.data.reshape(-1)
    o_flat = obs_data.data.reshape(-1)

    # We'll compute them in memory eventually, but let's define them step by step:
    # We must convert them to "event" 0/1 arrays. That means .compute().
    m_arr = (m_flat >= threshold).astype(int).compute()
    o_arr = (o_flat >= threshold).astype(int).compute()

    hits  = np.sum((m_arr == 1) & (o_arr == 1))
    misses= np.sum((m_arr == 0) & (o_arr == 1))
    fa    = np.sum((m_arr == 1) & (o_arr == 0))

    # POD
    pod = hits / (hits+misses) if (hits+misses) > 0 else np.nan
    # FAR
    far = fa / (hits+fa) if (hits+fa) > 0 else np.nan
    # CSI
    denom_csi = hits + misses + fa
    csi = hits/denom_csi if denom_csi > 0 else np.nan

    # ETS
    total = len(m_arr)
    correct_negatives = total - (hits + misses + fa)
    hits_rand = ((hits+misses)*(hits+fa))/total if total else 0
    ets_denom = (hits + misses + fa - hits_rand)
    if ets_denom == 0:
        ets = np.nan
    else:
        ets = (hits - hits_rand)/ets_denom

    # FSS (simplified fraction-based)
    # fraction of event in model vs. obs
    frac_model = np.mean(m_arr)
    frac_obs   = np.mean(o_arr)
    # basic approach
    fss_num = np.mean((m_arr - o_arr)**2)
    fss_den = np.mean(m_arr**2) + np.mean(o_arr**2)
    fss = 1.0 - fss_num/fss_den if fss_den != 0 else np.nan

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
    - Brier Score
    - Brier Skill Score (BSS)
    - (Placeholder) RPSS
    We'll do a final .compute() call.
    """
    p_arr = model_prob.data.reshape(-1)
    o_arr = obs_event.data.reshape(-1).astype(float)

    # Brier Score (BS)
    bs_dask = ((p_arr - o_arr)**2).mean()

    # If reference_prob is None, guess from mean of o_arr
    # That means "climatological" probability
    mean_obs_prob = o_arr.mean()
    # reference brier
    bs_ref_dask = ((mean_obs_prob - o_arr)**2).mean()

    bs_val, bs_ref_val = da.compute(bs_dask, bs_ref_dask)
    if bs_ref_val == 0:
        bss_val = np.nan
    else:
        bss_val = 1.0 - (bs_val/bs_ref_val)

    # Dummy RPSS
    rpss = np.nan

    return {
        'BS': bs_val,
        'BSS': bss_val,
        'RPSS': rpss
    }

########################
# 4. Distribution-Level
########################
def compute_distribution_metrics(model_data, obs_data, bins=50):
    """
    - Wasserstein distance
    - Jensen-Shannon
    We'll do .compute() to bring data locally, then calc with numpy/scipy.
    """
    # flatten
    m_arr_dask = model_data.data.reshape(-1)
    o_arr_dask = obs_data.data.reshape(-1)

    m_arr, o_arr = da.compute(m_arr_dask, o_arr_dask)

    # 1) Wasserstein
    wdist = wasserstein_distance(m_arr, o_arr)

    # 2) Jensen-Shannon
    min_val = min(np.min(m_arr), np.min(o_arr))
    max_val = max(np.max(m_arr), np.max(o_arr))

    # Hist
    m_hist, edges = np.histogram(m_arr, bins=bins, range=(min_val, max_val), density=True)
    o_hist, _     = np.histogram(o_arr, bins=bins, range=(min_val, max_val), density=True)

    # jensenshannon returns sqrt(JS)
    js_dist = jensenshannon(m_hist, o_hist)

    return {
        'Wasserstein': wdist,
        'JensenShannon': js_dist
    }

##############################################################################
#                        Visualization
##############################################################################

def visualize_map(model_data, obs_data, time_index=0):
    """
    Minimal cartopy-based plot for a single time step.
    We'll .compute() the slice for model & obs to load from Dask.
    """
    m_slice = model_data.isel(time=time_index).compute()
    o_slice = obs_data.isel(time=time_index).compute()

    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(16,5))

    ax1 = fig.add_subplot(1,2,1, projection=proj)
    ax1.set_title(f"Model (time={time_index})")
    ax1.coastlines()
    ax1.add_feature(cfeature.BORDERS, linestyle=':')
    m_slice.plot(
        ax=ax1, transform=proj,
        x='lon', y='lat', cmap='viridis',
        cbar_kwargs={"label": model_data.name}
    )

    ax2 = fig.add_subplot(1,2,2, projection=proj)
    ax2.set_title(f"Obs (time={time_index})")
    ax2.coastlines()
    ax2.add_feature(cfeature.BORDERS, linestyle=':')
    o_slice.plot(
        ax=ax2, transform=proj,
        x='lon', y='lat', cmap='viridis',
        cbar_kwargs={"label": obs_data.name}
    )

    plt.tight_layout()
    plt.show()

##############################################################################
#                             MAIN SCRIPT
##############################################################################

def main():
    print("=== Model vs. Observation Verification (Dask + GPU + Regrid) ===\n")

    # 1. Ask user if they want to use GPU
    use_gpu_input = input("Use GPU acceleration? (y/n) [default=n]: ").strip().lower()
    use_gpu = (use_gpu_input == 'y')
    if use_gpu and not GPU_AVAILABLE:
        print("GPU requested but CuPy not installed or GPU not available. Continuing on CPU.")
        use_gpu = False

    # 2. Number of workers
    workers_input = input("Number of Dask workers [default=1]: ").strip()
    n_workers = int(workers_input) if workers_input else 1

    # Initialize Dask
    client = init_dask_cluster(use_gpu=use_gpu, n_workers=n_workers)

    # 3. File paths
    model_path = input("Path to MODEL dataset: ").strip()
    obs_path   = input("Path to OBS dataset: ").strip()

    if not os.path.isfile(model_path):
        sys.exit(f"Model file not found: {model_path}")
    if not os.path.isfile(obs_path):
        sys.exit(f"Obs file not found: {obs_path}")

    # 4. Chunking
    chunk_input = input("Enter chunk size as 'time,lat,lon' [default=10,200,200]: ").strip()
    if chunk_input:
        try:
            tchunk, lchunk, lonchunk = chunk_input.split(',')
            chunks = {'time': int(tchunk), 'lat': int(lchunk), 'lon': int(lonchunk)}
        except:
            print("Invalid chunk format. Using default.")
            chunks = None
    else:
        chunks = None

    # 5. Open with chunking & optional GPU
    model_ds = open_dataset_with_chunks(model_path, chunks=chunks, use_gpu=use_gpu)
    obs_ds   = open_dataset_with_chunks(obs_path, chunks=chunks, use_gpu=use_gpu)

    # 6. Dimension check + regrid if needed
    try:
        model_ds, obs_ds = detect_or_fix_mismatch(model_ds, obs_ds)
    except (ValueError, KeyError) as e:
        sys.exit(f"Error in dimension check: {e}")

    # 7. Pick variable
    model_vars = list(model_ds.data_vars.keys())
    if not model_vars:
        sys.exit("No variables in model dataset.")
    print("\nAvailable model variables:")
    for i, v in enumerate(model_vars):
        print(f"[{i}] {v}")
    var_choice = input("Select variable by index or name: ").strip()
    if var_choice.isdigit():
        var_idx = int(var_choice)
        if var_idx < 0 or var_idx >= len(model_vars):
            sys.exit("Invalid index for variable.")
        variable_name = model_vars[var_idx]
    else:
        variable_name = var_choice
        if variable_name not in model_ds.data_vars:
            sys.exit(f"Variable '{variable_name}' not in model dataset.")

    # Check if exists in obs
    if variable_name not in obs_ds.data_vars:
        sys.exit(f"Variable '{variable_name}' not found in obs dataset: {list(obs_ds.data_vars.keys())}")

    model_da = model_ds[variable_name]
    obs_da   = obs_ds[variable_name]

    # 8. Visualization
    time_index_input = input("\nTime index to visualize [default=0]: ").strip()
    if time_index_input == '':
        time_index = 0
    else:
        time_index = int(time_index_input)
    try:
        visualize_map(model_da, obs_da, time_index=time_index)
    except Exception as ex:
        print(f"Visualization error: {ex}")

    # 9. Which metrics to compute
    print("\nWhich metrics do you want to compute? (You can select multiple)")
    print("1) Continuous (MSE, MAE, Bias, Corr)")
    print("2) Event-based (POD, FAR, CSI, ETS, FSS)")
    print("3) Probabilistic (Brier Score, BSS, RPSS)")
    print("4) Distribution-level (Wasserstein, Jensen-Shannon)")
    selection = input("Enter e.g. '1,2': ").strip()
    metric_choices = [s.strip() for s in selection.split(',') if s.strip()]

    # 9.1 Continuous
    if '1' in metric_choices:
        cont_res = compute_continuous_metrics(model_da, obs_da)
        print("\n-- Continuous Metrics --")
        for k,v in cont_res.items():
            print(f"{k}: {v:.4f}")

    # 9.2 Event-based
    if '2' in metric_choices:
        thr_str = input("Enter threshold for event-based metrics [default=1.0]: ").strip()
        threshold = float(thr_str) if thr_str else 1.0
        evt_res = compute_event_metrics(model_da, obs_da, threshold=threshold)
        print("\n-- Event-Based Metrics --")
        for k,v in evt_res.items():
            if isinstance(v, float):
                print(f"{k}: {v:.4f}")
            else:
                print(f"{k}: {v}")

    # 9.3 Probabilistic
    if '3' in metric_choices:
        # We assume model_da is probability (0..1), obs_da is 0/1
        # or we ask user if we should threshold the obs
        prob_warning = input("Is the model data actually probabilities (y/n)? [default=n]: ").strip().lower()
        if prob_warning == 'y':
            obs_thr_str = input("Threshold to convert obs->0/1 [default=0.5]: ").strip()
            obs_threshold = float(obs_thr_str) if obs_thr_str else 0.5
            obs_event_da = (obs_da >= obs_threshold).astype(int)
            prob_res = compute_probabilistic_metrics(model_da, obs_event_da)
            print("\n-- Probabilistic Metrics --")
            for k,v in prob_res.items():
                if isinstance(v,float):
                    print(f"{k}: {v:.4f}")
                else:
                    print(f"{k}: {v}")
        else:
            print("Skipping probabilistic metrics since data not treated as probabilities.\n")

    # 9.4 Distribution-level
    if '4' in metric_choices:
        dist_res = compute_distribution_metrics(model_da, obs_da)
        print("\n-- Distribution Metrics --")
        for k,v in dist_res.items():
            print(f"{k}: {v:.4f}")

    print("\nAll done! You can monitor performance in the Dask dashboard for large computations.")
    client.close()

if __name__ == "__main__":
    main()
