"""
veriftool.backend
=================
This module provides functions to compare a climate model vs. observation data:
 - Dask cluster setup (CPU/GPU)
 - NetCDF loading with chunking
 - Regridding / dimension checks
 - Domain-wide metrics (continuous, event-based, probabilistic, distribution-level)
 - Per-pixel (gridwise) metrics for bias maps, correlation maps, event detection maps
 - Basic cartopy-based visualization functions

Example usage (within Python):

    from veriftool.backend import (
        init_dask_cluster,
        open_dataset_with_chunks,
        detect_or_fix_mismatch,
        compute_continuous_metrics,
        # ...
    )

    client = init_dask_cluster(use_gpu=False, n_workers=2)
    model_ds = open_dataset_with_chunks("path/to/model.nc", chunks={'time':10,'lat':200,'lon':200})
    obs_ds   = open_dataset_with_chunks("path/to/obs.nc")
    model_ds, obs_ds = detect_or_fix_mismatch(model_ds, obs_ds)
    # Then pick variables and compute your metrics, etc.

"""

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
import streamlit as st

# Attempt to import CuPy for GPU usage
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

from dask.distributed import Client, LocalCluster


##############################################################################
# 1. DASK & GPU SETUP
##############################################################################

def init_dask_cluster(use_gpu: bool = False, n_workers: int = 1):
    """
    Initializes a local Dask cluster for parallel computation.

    Parameters
    ----------
    use_gpu : bool
        If True and CuPy is installed, attempts to create a GPU-based cluster.
    n_workers : int
        Number of workers in the local cluster.

    Returns
    -------
    client : dask.distributed.Client
        A Dask client connected to the newly created cluster.
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

    Parameters
    ----------
    path : str
        Path to the NetCDF file.
    chunks : dict, optional
        Dask chunk sizes, e.g. {'time':10, 'lat':200, 'lon':200}.
    use_gpu : bool
        If True, attempt CuPy-based arrays.

    Returns
    -------
    ds : xarray.Dataset
        The opened and chunked dataset.
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
# 2. REGRID & DIMENSION CHECK
##############################################################################

def detect_or_fix_mismatch(model_ds, obs_ds, required_dims=('time','lat','lon')):
    """
    Checks dimension mismatch for the specified dims (time, lat, lon).
    If mismatch is detected, interactively prompts user to regrid or exit.

    Parameters
    ----------
    model_ds : xarray.Dataset
        The model dataset.
    obs_ds : xarray.Dataset
        The observation dataset.
    required_dims : tuple
        Dimensions to check for consistency (time, lat, lon).

    Returns
    -------
    model_ds, obs_ds : xarray.Dataset
        Potentially regridded datasets that match in size/coords.
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
# 3. PER-TIME AGGREGATE METRICS
##############################################################################

def compute_continuous_metrics(model_data, obs_data):
    """
    Computes domain-wide MSE, MAE, Bias, and Correlation (flattened).

    Returns a dictionary of scalar metrics:
      - MSE
      - MAE
      - Bias
      - Corr
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
    Domain-wide event-based metrics (flattened):
      - POD
      - FAR
      - CSI
      - ETS
      - FSS (placeholder, simplified)

    `threshold` defines what counts as an event: data >= threshold -> 1, else 0.
    """
    m_flat = model_data.data.reshape(-1)
    o_flat = obs_data.data.reshape(-1)

    # Convert to 0/1
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

    # Simplified fraction skill score
    fss_num = np.mean((m_arr - o_arr)**2)
    fss_den = np.mean(m_arr**2) + np.mean(o_arr**2)
    fss = 1 - fss_num/fss_den if fss_den!=0 else np.nan

    return {
        'POD': pod,
        'FAR': far,
        'CSI': csi,
        'ETS': ets,
        'FSS': fss
    }


def compute_probabilistic_metrics(model_prob, obs_event):
    """
    Domain-wide probabilistic metrics:
      - Brier Score (BS)
      - Brier Skill Score (BSS)
      - RPSS (placeholder, currently NaN)

    `model_prob` is a DataArray of predicted probabilities (0..1).
    `obs_event` is a 0/1 array for whether event occurred.
    """
    p_arr = model_prob.data.reshape(-1)
    o_arr = obs_event.data.reshape(-1).astype(float)

    bs_dask = ((p_arr - o_arr)**2).mean()
    bs_val = bs_dask.compute()

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
    Domain-wide distribution-based metrics:
      - Wasserstein distance
      - Jensen-Shannon divergence

    Hist-based approach for JS. For large data, might need to chunk carefully.
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

    return {
        'Wasserstein': wdist,
        'JensenShannon': js_dist
    }


##############################################################################
# 4. PER-PIXEL (GRIDWISE) METRICS ACROSS TIME
##############################################################################

def compute_gridwise_continuous(model_da, obs_da):
    """
    Returns an xarray.Dataset with:
      - bias_map, mae_map, mse_map, corr_map
    Each is (lat, lon) computed over the time dimension.
    """
    bias_map = (model_da - obs_da).mean(dim='time')
    mae_map  = np.abs(model_da - obs_da).mean(dim='time')
    mse_map  = ((model_da - obs_da)**2).mean(dim='time')

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
    ds.attrs["description"] = "Per-pixel continuous metrics (time dimension collapsed)."
    return ds


def compute_gridwise_event_metrics(model_da, obs_da, threshold=1.0):
    """
    Returns an xarray.Dataset with:
      - POD_map, FAR_map, CSI_map, ETS_map

    Threshold-based event detection across time dimension.
    """
    model_event = (model_da >= threshold).astype(int)
    obs_event   = (obs_da   >= threshold).astype(int)

    hits  = (model_event * obs_event).sum(dim='time')
    mod_sum = model_event.sum(dim='time')
    obs_sum = obs_event.sum(dim='time')
    misses = obs_sum - hits
    fa     = mod_sum - hits

    hits_f   = hits.astype(float)
    misses_f = misses.astype(float)
    fa_f     = fa.astype(float)

    time_length = model_da.sizes['time']

    pod_map = hits_f / (hits_f + misses_f)
    far_map = fa_f   / (hits_f + fa_f)
    csi_map = hits_f / (hits_f + misses_f + fa_f)

    hits_rand = (obs_sum * mod_sum) / float(time_length)
    ets_denom = (hits_f + misses_f + fa_f - hits_rand)
    ets_map   = (hits_f - hits_rand) / ets_denom

    ds = xr.Dataset({
        "POD_map": pod_map,
        "FAR_map": far_map,
        "CSI_map": csi_map,
        "ETS_map": ets_map,
    })
    ds.attrs["description"] = f"Gridwise event-based metrics, threshold={threshold}"
    return ds


##############################################################################
# 5. VISUALIZATION FUNCTIONS (FIXED FOR STREAMLIT)
##############################################################################

def visualize_map(model_data, obs_data, time_index=0):
    """
    Side-by-side visualization of model vs. observation at a single time step.
    """
    m_slice = model_data.isel(time=time_index).compute()
    o_slice = obs_data.isel(time=time_index).compute()

    fig = plt.figure(figsize=(12, 5))

    # Model Plot
    ax1 = fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree())
    ax1.set_title(f"Model (time={time_index})")
    ax1.coastlines()
    ax1.add_feature(cfeature.BORDERS, linestyle=':')
    m_slice.plot(ax=ax1, transform=ccrs.PlateCarree(), x='lon', y='lat', cmap='viridis', cbar_kwargs={"label": model_data.name})

    # Observation Plot
    ax2 = fig.add_subplot(1, 2, 2, projection=ccrs.PlateCarree())
    ax2.set_title(f"Obs (time={time_index})")
    ax2.coastlines()
    ax2.add_feature(cfeature.BORDERS, linestyle=':')
    o_slice.plot(ax=ax2, transform=ccrs.PlateCarree(), x='lon', y='lat', cmap='viridis', cbar_kwargs={"label": obs_data.name})

    plt.tight_layout()
    st.pyplot(fig)  # ✅ FIXED FOR STREAMLIT


def visualize_metric_map(metric_da, title_str="", cmap="viridis", center=None):
    """
    Visualization for per-pixel metrics like Bias, MAE, MSE, and Correlation.
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    ax.set_title(title_str)
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':')

    kwargs = {"x": "lon", "y": "lat", "cmap": cmap, "transform": ccrs.PlateCarree(), "cbar_kwargs": {"label": metric_da.name}}
    if center is not None:
        kwargs["center"] = center

    metric_da.plot(ax=ax, **kwargs)

    plt.tight_layout()
    st.pyplot(fig)  # ✅ FIXED FOR STREAMLIT