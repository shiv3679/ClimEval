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
    Initializes a local Dask cluster (CPU or GPU).
    """
    if use_gpu:
        print("Initializing Dask LocalCluster for GPU usage...")
        cluster = LocalCluster(
            n_workers=n_workers,
            threads_per_worker=1,  # typically 1 thread per GPU
            processes=False
        )
    else:
        print("Initializing Dask LocalCluster for CPU usage...")
        cluster = LocalCluster(
            n_workers=n_workers,
            threads_per_worker=max(1, os.cpu_count() // n_workers if n_workers > 0 else 1)
        )
    client = Client(cluster)
    print(f"Dask dashboard link: {client.dashboard_link}\n")
    return client


def open_dataset_with_chunks(path, chunks=None, use_gpu=False):
    """
    Opens a NetCDF file with xarray, applying Dask chunking.
    If use_gpu=True and CuPy is available, convert blocks to CuPy arrays.
    """
    if chunks is None:
        chunks = {'time': 10, 'lat': 200, 'lon': 200}

    print(f"Opening dataset at {path} with chunks={chunks}")
    ds = xr.open_dataset(path, chunks=chunks)

    if use_gpu and GPU_AVAILABLE:
        print("Mapping dataset to CuPy arrays...")
        for var in ds.data_vars:
            ds[var].data = ds[var].data.map_blocks(cp.asarray, dtype=ds[var].dtype)
    elif use_gpu and not GPU_AVAILABLE:
        print("GPU requested but CuPy not available; continuing on CPU.")

    return ds

##############################################################################
#                REGRID & DIMENSION CHECK
##############################################################################

def detect_or_fix_mismatch(model_ds, obs_ds, required_dims=('time','lat','lon')):
    """
    Checks dimension mismatch. If mismatch, user can choose to regrid or exit.
    Returns updated (model_ds, obs_ds).
    """
    for dim in required_dims:
        if dim not in model_ds.dims:
            raise ValueError(f"Model dataset missing dim: {dim}")
        if dim not in obs_ds.dims:
            raise ValueError(f"Obs dataset missing dim: {dim}")

    same_shape = True
    for dim in required_dims:
        if model_ds.sizes[dim] != obs_ds.sizes[dim]:
            same_shape = False
            break

    if same_shape:
        print("Datasets have same dimension sizes. Checking coordinate arrays.")
        mod_lat = model_ds['lat'].values
        mod_lon = model_ds['lon'].values
        mod_time= model_ds['time'].values

        obs_lat = obs_ds['lat'].values
        obs_lon = obs_ds['lon'].values
        obs_time= obs_ds['time'].values

        coords_match = True
        if not np.allclose(mod_lat, obs_lat):
            coords_match = False
        if not np.allclose(mod_lon, obs_lon):
            coords_match = False
        if not np.array_equal(mod_time, obs_time):
            coords_match = False

        if coords_match:
            print("Dimension sizes & coords match. No regridding needed.\n")
            return model_ds, obs_ds
        else:
            print("Coord arrays differ.\n")
    else:
        print("Dimension sizes differ.\n")

    choice = input("Mismatch found. Regrid? (1) Regrid or (2) Exit [default=2]: ").strip()
    if choice == '1':
        direction = input("Regrid model->obs (m2o) or obs->model (o2m)? [default=m2o]: ").strip().lower()
        if direction == 'o2m':
            print("Regridding Obs -> Model coords...")
            obs_ds = obs_ds.interp_like(model_ds, method='linear')
        else:
            print("Regridding Model -> Obs coords...")
            model_ds = model_ds.interp_like(obs_ds, method='linear')
        print("Regridding done.\n")
        return model_ds, obs_ds
    else:
        sys.exit("Exiting; please manually align your data.")


##############################################################################
#                   PER-TIME AGGREGATE METRICS
##############################################################################

def compute_continuous_metrics(model_data, obs_data):
    """
    Domain-wide MSE, MAE, Bias, Corr (flatten over space,time).
    Dask-based, returns scalar results.
    """
    m_arr = model_data.data.reshape(-1)
    o_arr = obs_data.data.reshape(-1)

    mse_dask = ((m_arr - o_arr)**2).mean()
    mae_dask = da.fabs(m_arr - o_arr).mean()
    bias_dask= (m_arr - o_arr).mean()

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


def compute_event_metrics(model_data, obs_data, threshold=1.0):
    """
    Domain-wide event-based metrics: POD, FAR, CSI, ETS, FSS (flatten).
    """
    m_flat = model_data.data.reshape(-1)
    o_flat = obs_data.data.reshape(-1)

    # Convert to 0/1 (requires .compute() eventually):
    m_arr = (m_flat >= threshold).astype(int).compute()
    o_arr = (o_flat >= threshold).astype(int).compute()

    hits  = np.sum((m_arr == 1) & (o_arr == 1))
    misses= np.sum((m_arr == 0) & (o_arr == 1))
    fa    = np.sum((m_arr == 1) & (o_arr == 0))

    pod = hits/(hits+misses) if (hits+misses)>0 else np.nan
    far = fa/(hits+fa) if (hits+fa)>0 else np.nan
    csi = hits/(hits+misses+fa) if (hits+misses+fa)>0 else np.nan

    total = len(m_arr)
    hits_rand = ((hits+misses)*(hits+fa))/total if total else 0
    ets_denom = (hits + misses + fa - hits_rand)
    if ets_denom == 0:
        ets = np.nan
    else:
        ets = (hits - hits_rand)/ets_denom

    # Simplified FSS
    fss_num = np.mean((m_arr - o_arr)**2)
    fss_den = np.mean(m_arr**2) + np.mean(o_arr**2)
    fss = 1 - fss_num/fss_den if fss_den!=0 else np.nan

    return {
        'POD': pod, 'FAR': far, 'CSI': csi, 'ETS': ets, 'FSS': fss
    }


def compute_probabilistic_metrics(model_prob, obs_event):
    """
    Domain-wide Brier Score, BSS, placeholder RPSS.
    """
    p_arr = model_prob.data.reshape(-1)
    o_arr = obs_event.data.reshape(-1).astype(float)

    bs_dask = ((p_arr - o_arr)**2).mean()
    bs_val = bs_dask.compute()

    # reference
    mean_obs = np.mean(o_arr)
    bs_ref = np.mean((mean_obs - o_arr)**2)
    bss_val = np.nan if bs_ref==0 else 1 - bs_val/bs_ref

    return {
        'BS': bs_val,
        'BSS': bss_val,
        'RPSS': np.nan
    }


def compute_distribution_metrics(model_data, obs_data, bins=50):
    """
    Domain-wide distribution metrics: 
    - Wasserstein distance
    - Jensen-Shannon divergence
    """
    m_arr_dask = model_data.data.reshape(-1)
    o_arr_dask = obs_data.data.reshape(-1)
    m_arr, o_arr = da.compute(m_arr_dask, o_arr_dask)

    wdist = wasserstein_distance(m_arr, o_arr)
    min_val = min(m_arr.min(), o_arr.min())
    max_val = max(m_arr.max(), o_arr.max())

    m_hist, edges = np.histogram(m_arr, bins=bins, range=(min_val, max_val), density=True)
    o_hist, _     = np.histogram(o_arr, bins=bins, range=(min_val, max_val), density=True)
    js_dist = jensenshannon(m_hist, o_hist)

    return {'Wasserstein': wdist, 'JensenShannon': js_dist}


##############################################################################
#               PER-PIXEL (GRIDWISE) METRICS ACROSS TIME
##############################################################################

def compute_gridwise_continuous(model_da, obs_da):
    """
    Returns an xarray.Dataset with:
      - 'bias_map' (lat, lon)
      - 'mae_map'  (lat, lon)
      - 'mse_map'  (lat, lon)
      - 'corr_map' (lat, lon) [corr over the time dimension]
    This allows the user to see how these metrics vary spatially.
    """
    bias_map = (model_da - obs_da).mean(dim='time')
    mae_map  = np.abs(model_da - obs_da).mean(dim='time')  # Use np.abs
    mse_map  = ((model_da - obs_da)**2).mean(dim='time')

    # correlation across time dimension
    model_mean = model_da.mean(dim='time')
    obs_mean   = obs_da.mean(dim='time')
    model_anom = model_da - model_mean
    obs_anom   = obs_da - obs_mean
    cov        = (model_anom * obs_anom).mean(dim='time')
    std_m      = model_da.std(dim='time')
    std_o      = obs_da.std(dim='time')
    corr_map   = cov / (std_m * std_o)

    ds = xr.Dataset({
        "bias_map": bias_map,
        "mae_map": mae_map,
        "mse_map": mse_map,
        "corr_map": corr_map
    })
    ds.attrs["description"] = "Per-pixel continuous metrics across time."
    return ds


def compute_gridwise_event_metrics(model_da, obs_da, threshold=1.0):
    """
    Returns an xarray.Dataset with:
      - 'POD_map' (lat, lon)
      - 'FAR_map' (lat, lon)
      - 'CSI_map' (lat, lon)
      - 'ETS_map' (lat, lon)

    For each pixel, we threshold the model and obs, then sum
    hits, misses, false alarms over time.
    """
    model_event = (model_da >= threshold).astype(int)
    obs_event   = (obs_da   >= threshold).astype(int)

    # Summation across time => hits, misses, false alarms
    hits  = (model_event * obs_event).sum(dim='time')
    mod_sum = model_event.sum(dim='time')
    obs_sum = obs_event.sum(dim='time')
    misses = obs_sum - hits
    fa     = mod_sum - hits

    hits_f   = hits.astype(float)
    misses_f = misses.astype(float)
    fa_f     = fa.astype(float)

    time_length = model_da.sizes['time']

    # Probability of Detection (POD)
    pod_map = hits_f / (hits_f + misses_f)

    # False Alarm Ratio (FAR)
    far_map = fa_f / (hits_f + fa_f)

    # CSI
    csi_map = hits_f / (hits_f + misses_f + fa_f)

    # ETS
    hits_rand = (obs_sum * mod_sum) / float(time_length)
    ets_denom = (hits_f + misses_f + fa_f - hits_rand)
    ets_map   = (hits_f - hits_rand) / ets_denom

    ds = xr.Dataset({
        "POD_map": pod_map,
        "FAR_map": far_map,
        "CSI_map": csi_map,
        "ETS_map": ets_map,
    })
    ds.attrs["description"] = f"Gridwise event-based metrics (threshold={threshold})."
    return ds


##############################################################################
#                           VISUALIZATION
##############################################################################

def visualize_map(model_data, obs_data, time_index=0):
    """
    Just a quick side-by-side plot of model vs. obs at a single time slice.
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


def visualize_metric_map(metric_da, title_str="", cmap="viridis", center=None):
    """
    Plots a single (lat, lon) DataArray using Cartopy.
    If 'center' is not None, we might use a diverging colormap with center=0.
    """
    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(8,6))
    ax  = fig.add_subplot(1,1,1, projection=proj)
    ax.set_title(title_str)
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':')

    kwargs = {
        "x": "lon",
        "y": "lat",
        "cmap": cmap,
        "transform": proj,
        "cbar_kwargs": {"label": metric_da.name}
    }
    if center is not None:
        # e.g. center=0 for bias, correlation
        kwargs["center"] = center

    metric_da.plot(ax=ax, **kwargs)
    plt.tight_layout()
    plt.show()


##############################################################################
#                             MAIN
##############################################################################

def main():
    print("=== Model vs. Observation Verification (Per-Pixel Metrics) ===\n")

    # 1. GPU usage
    use_gpu_input = input("Use GPU acceleration? (y/n) [default=n]: ").strip().lower()
    use_gpu = (use_gpu_input == 'y')
    if use_gpu and not GPU_AVAILABLE:
        print("GPU requested but CuPy not found; continuing on CPU.")
        use_gpu = False

    # 2. Dask workers
    worker_str = input("Number of Dask workers [default=1]: ").strip()
    n_workers = int(worker_str) if worker_str else 1
    client = init_dask_cluster(use_gpu=use_gpu, n_workers=n_workers)

    # 3. Paths
    model_path = input("Path to MODEL dataset: ").strip()
    obs_path   = input("Path to OBS dataset: ").strip()
    if not os.path.isfile(model_path):
        sys.exit(f"Model file not found: {model_path}")
    if not os.path.isfile(obs_path):
        sys.exit(f"Obs file not found: {obs_path}")

    # 4. Chunking
    chunk_str = input("Chunk sizes 'time,lat,lon' [default=10,200,200]: ").strip()
    if chunk_str:
        try:
            tchunk, lchunk, lochunk = chunk_str.split(',')
            chunks = {'time': int(tchunk), 'lat': int(lchunk), 'lon': int(lochunk)}
        except:
            print("Invalid chunk format; using default.")
            chunks = None
    else:
        chunks = None

    model_ds = open_dataset_with_chunks(model_path, chunks=chunks, use_gpu=use_gpu)
    obs_ds   = open_dataset_with_chunks(obs_path,   chunks=chunks, use_gpu=use_gpu)

    # 5. Dimension check / regrid
    try:
        model_ds, obs_ds = detect_or_fix_mismatch(model_ds, obs_ds)
    except (ValueError, KeyError) as e:
        sys.exit(f"Dimension mismatch error: {e}")

    # 6. Pick variable
    model_vars = list(model_ds.data_vars.keys())
    if not model_vars:
        sys.exit("No variables in model dataset.")
    print("\nAvailable model variables:")
    for i,v in enumerate(model_vars):
        print(f"[{i}] {v}")
    var_choice = input("Select variable index or name: ").strip()
    if var_choice.isdigit():
        idx = int(var_choice)
        if idx < 0 or idx >= len(model_vars):
            sys.exit("Invalid variable index.")
        variable_name = model_vars[idx]
    else:
        variable_name = var_choice

    if variable_name not in model_ds.data_vars:
        sys.exit(f"Variable '{variable_name}' not in model dataset.")
    if variable_name not in obs_ds.data_vars:
        sys.exit(f"Variable '{variable_name}' not in obs dataset.")
    
    model_da = model_ds[variable_name]
    obs_da   = obs_ds[variable_name]

    # 7. Quick visualization
    t_str = input("\nTime index to visualize [default=0]: ").strip()
    time_index = int(t_str) if t_str else 0
    try:
        visualize_map(model_da, obs_da, time_index=time_index)
    except Exception as ex:
        print("Visualization error:", ex)

    # 8. Basic domain-wide metrics
    metric_str = input("\nWhich domain-wide metrics do you want? (1=Cont,2=Event,3=Prob,4=Dist) e.g. '1,2': ").strip()
    picks = [x.strip() for x in metric_str.split(',') if x.strip()]
    # Continuous
    if '1' in picks:
        cont = compute_continuous_metrics(model_da, obs_da)
        print("\n-- Domain-wide Continuous Metrics --")
        for k,v in cont.items():
            print(f"{k}: {v:.4f}")
    # Event
    if '2' in picks:
        thr_str = input("Threshold for event-based [default=1.0]: ").strip()
        threshold = float(thr_str) if thr_str else 1.0
        evt = compute_event_metrics(model_da, obs_da, threshold=threshold)
        print("\n-- Domain-wide Event Metrics --")
        for k,v in evt.items():
            print(f"{k}: {v:.4f}" if isinstance(v,float) else f"{k}: {v}")
    # Prob
    if '3' in picks:
        prob_q = input("Is model data probability? (y/n) [default=n]: ").strip().lower()
        if prob_q=='y':
            obs_thr_s = input("Obs threshold for 0/1 [default=0.5]: ").strip()
            obs_thr = float(obs_thr_s) if obs_thr_s else 0.5
            obs_ev   = (obs_da >= obs_thr).astype(int)
            prob_res = compute_probabilistic_metrics(model_da, obs_ev)
            print("\n-- Domain-wide Probabilistic Metrics --")
            for k,v in prob_res.items():
                print(f"{k}: {v:.4f}" if isinstance(v,float) else f"{k}: {v}")
        else:
            print("Skipping prob metrics; data not probability.")
    # Dist
    if '4' in picks:
        dist = compute_distribution_metrics(model_da, obs_da)
        print("\n-- Domain-wide Distribution Metrics --")
        for k,v in dist.items():
            print(f"{k}: {v:.4f}")

    # 9. Per-pixel (gridwise) metrics across time
    print("\nDo you want per-pixel metrics (2D lat-lon maps) over time? (y/n) [default=n]")
    gridwise_ans = input().strip().lower()
    if gridwise_ans == 'y':
        # A) Continuous
        print("Calculating gridwise continuous metrics... (bias, mae, mse, corr)")
        grid_cont = compute_gridwise_continuous(model_da, obs_da).compute()
        # For demonstration, let's just plot each map
        visualize_metric_map(grid_cont["bias_map"],  title_str="Bias Map",  cmap="RdBu", center=0)
        visualize_metric_map(grid_cont["mae_map"],   title_str="MAE Map",   cmap="Reds")
        visualize_metric_map(grid_cont["mse_map"],   title_str="MSE Map",   cmap="Reds")
        visualize_metric_map(grid_cont["corr_map"],  title_str="Correlation Map", cmap="RdBu", center=0)

        # B) Event-based
        thr_str = input("\nThreshold for gridwise event-based metrics? [default=1.0]: ").strip()
        threshold = float(thr_str) if thr_str else 1.0
        print("Calculating gridwise event-based metrics... (POD, FAR, CSI, ETS)")
        grid_evt = compute_gridwise_event_metrics(model_da, obs_da, threshold=threshold).compute()

        # Visualize
        visualize_metric_map(grid_evt["POD_map"], title_str="POD Map", cmap="Blues")
        visualize_metric_map(grid_evt["FAR_map"], title_str="FAR Map", cmap="Reds")
        visualize_metric_map(grid_evt["CSI_map"], title_str="CSI Map", cmap="Greens")
        visualize_metric_map(grid_evt["ETS_map"], title_str="ETS Map", cmap="RdBu", center=0)

        # Optionally save to NetCDF
        # grid_cont.to_netcdf("gridwise_continuous_metrics.nc")
        # grid_evt.to_netcdf("gridwise_event_metrics.nc")

    print("\nDone! Monitor Dask dashboard for performance details.")
    client.close()


if __name__=="__main__":
    main()
