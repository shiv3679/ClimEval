#!/usr/bin/env python3

import os
import sys
import math
import numpy as np
import xarray as xr
import dask.array as da
import matplotlib.pyplot as plt

# Cartopy for map plots (same as before)
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Dask distributed for parallelism
# (You can also use dask-cuda or dask_jobqueue, etc. for HPC or GPU setups)
from dask.distributed import Client, LocalCluster

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    # CuPy not installed or GPU not available
    GPU_AVAILABLE = False


############################################################
#                   HELPER FUNCTIONS
############################################################

def init_dask_cluster(use_gpu: bool = False, n_workers: int = 1):
    """
    Initializes a local Dask cluster (either CPU or GPU).
    For large HPC jobs, you might configure differently.
    """
    if use_gpu:
        print("Initializing Dask LocalCluster for GPU usage...")
        # If you have multiple GPUs, you can scale n_workers accordingly.
        # In a real HPC environment, you might use dask-cuda or dask-mpi.
        cluster = LocalCluster(
            n_workers=n_workers,
            threads_per_worker=1,  # Typically 1 thread per GPU device
            processes=False
        )
    else:
        print("Initializing Dask LocalCluster for CPU usage...")
        cluster = LocalCluster(
            n_workers=n_workers,
            threads_per_worker=os.cpu_count() // n_workers if n_workers > 0 else 1
        )
    client = Client(cluster)
    print(f"Dask dashboard available at: {client.dashboard_link}")
    return client


def open_dataset_with_chunks(path, chunks=None, use_gpu=False):
    """
    Opens a NetCDF file with xarray, applying Dask chunking.
    If use_gpu=True, we convert arrays to CuPy (when possible).
    """
    if chunks is None:
        # Reasonable default chunks (adjust as needed)
        chunks = {'time': 10, 'lat': 200, 'lon': 200}

    print(f"Opening dataset at {path} with chunks={chunks}")
    ds = xr.open_dataset(path, chunks=chunks)

    if use_gpu and GPU_AVAILABLE:
        print("Attempting to convert dataset to GPU (CuPy) arrays...")
        # We can try to map all DataArray chunks to cupy
        # (But be aware some operations may not yet be GPU-accelerated)
        # We'll do a shallow copy of the dataset with each variable mapped to CuPy.
        for var in ds.data_vars:
            ds[var].data = ds[var].data.map_blocks(cp.asarray, dtype=ds[var].dtype)
    else:
        if use_gpu and not GPU_AVAILABLE:
            print("Warning: use_gpu=True requested, but CuPy not available. Continuing on CPU.")
    
    return ds


def validate_variable_existence(ds, var_name):
    if var_name not in ds.data_vars:
        raise KeyError(f"Variable '{var_name}' not found in dataset. "
                       f"Available variables: {list(ds.data_vars.keys())}")


def detect_or_fix_mismatch(model_ds, obs_ds, var_name):
    """
    Checks dimension mismatch in (time, lat, lon). If mismatch is found,
    prompts user to regrid or exit. Uses .interp_like() with dask, which
    can also run on GPU if chunked properly.
    """
    required_dims = ['time', 'lat', 'lon']
    for dim in required_dims:
        if dim not in model_ds.dims:
            raise ValueError(f"Model dataset missing dimension: {dim}")
        if dim not in obs_ds.dims:
            raise ValueError(f"Obs dataset missing dimension: {dim}")
    
    same_shape = True
    for dim in required_dims:
        if model_ds.sizes[dim] != obs_ds.sizes[dim]:
            same_shape = False
            break
    
    if same_shape:
        # Check coordinate arrays
        # Because these are dask arrays, we typically load them (compute) 
        # to check equivalence. For large data, watch out for memory usage.
        model_lat = model_ds['lat'].values
        obs_lat   = obs_ds['lat'].values
        model_lon = model_ds['lon'].values
        obs_lon   = obs_ds['lon'].values
        model_time = model_ds['time'].values
        obs_time   = obs_ds['time'].values

        coords_match = True
        if not np.allclose(model_lat, obs_lat):
            coords_match = False
        if not np.allclose(model_lon, obs_lon):
            coords_match = False
        if not np.array_equal(model_time, obs_time):
            coords_match = False
        
        if coords_match:
            print("Dataset dimensions and coordinates match—no regridding needed.\n")
            return model_ds, obs_ds
        else:
            print("Dimension sizes match, but coordinate values differ.")
    else:
        print("Mismatch in dimension sizes (time, lat, or lon).")
    
    # We have a mismatch
    user_choice = input(
        "Mismatch detected. Would you like to (1) regrid or (2) exit?\n"
        "Enter '1' or '2': "
    ).strip()
    if user_choice == '2':
        sys.exit("Exiting. Please reformat your data manually.")
    elif user_choice != '1':
        sys.exit("Invalid selection. Exiting.")
    
    regrid_dir = input(
        "Enter 'm2o' to regrid Model->Obs, or 'o2m' to regrid Obs->Model: "
    ).strip().lower()

    if regrid_dir == 'm2o':
        print("Regridding Model -> Observation coordinates ...")
        model_ds = model_ds.interp_like(obs_ds, method='linear')
    elif regrid_dir == 'o2m':
        print("Regridding Obs -> Model coordinates ...")
        obs_ds = obs_ds.interp_like(model_ds, method='linear')
    else:
        sys.exit("Invalid regrid option. Exiting.")
    
    print("Regridding complete!\n")
    return model_ds, obs_ds


def check_missing_values(ds, var_name):
    """
    Checks for missing values (NaNs). Because ds[var_name] might be a Dask array,
    we use .any().compute().
    """
    data = ds[var_name].data
    return da.isnan(data).any().compute()  # triggers Dask compute


def extended_global_stats(model_da, obs_da):
    """
    Compute domain-wide metrics lazily using dask, then .compute()
    so everything runs out-of-core or on GPU if available:
      - bias, mae, rmse, correlation, std_model, std_obs, std_ratio, nse
    """
    # Flatten to 1D
    model_arr = model_da.data.reshape(-1)
    obs_arr   = obs_da.data.reshape(-1)

    # Means
    mean_model = model_arr.mean()
    mean_obs   = obs_arr.mean()

    # Differences
    diff = model_arr - obs_arr
    bias = diff.mean()
    mae  = da.fabs(diff).mean()
    mse  = (diff**2).mean()  # for RMSE
    rmse = da.sqrt(mse)

    # Correlation (Pearson)
    # Corr = cov(X, Y) / (std(X) * std(Y))
    cov = ((model_arr - mean_model) * (obs_arr - mean_obs)).mean()
    std_m = da.sqrt(((model_arr - mean_model)**2).mean())
    std_o = da.sqrt(((obs_arr - mean_obs)**2).mean())
    corr = cov / (std_m * std_o)

    # ratio of stdev
    std_ratio = std_m / std_o

    # NSE: 1 - (sum((m - o)^2)/sum((o - mean(o))^2))
    nse_num = (diff**2).sum()
    nse_den = ((obs_arr - mean_obs)**2).sum()
    nse     = 1.0 - (nse_num / nse_den)

    # We now compute everything in one go:
    (bias_val, mae_val, rmse_val, corr_val,
     std_m_val, std_o_val, std_ratio_val, nse_val) = da.compute(
        bias, mae, rmse, corr, std_m, std_o, std_ratio, nse
    )

    return {
        'bias': bias_val,
        'mae': mae_val,
        'rmse': rmse_val,
        'corr': corr_val,
        'std_model': std_m_val,
        'std_obs': std_o_val,
        'std_ratio': std_ratio_val,
        'nse': nse_val
    }


def compute_gridwise_metrics(model_da, obs_da):
    """
    Per-gridpoint metrics (difference, abs_difference, correlation).
    We'll keep them as dask arrays and only finalize with .compute() upon saving or plotting.
    """
    difference = model_da - obs_da
    abs_difference = abs(difference)

    model_mean = model_da.mean(dim='time')
    obs_mean   = obs_da.mean(dim='time')

    model_anom = model_da - model_mean
    obs_anom   = obs_da - obs_mean

    cov = (model_anom * obs_anom).mean(dim='time')
    model_std = model_da.std(dim='time')
    obs_std   = obs_da.std(dim='time')
    corr = cov / (model_std * obs_std)

    metrics_ds = xr.Dataset(
        {
            "difference": difference,
            "abs_difference": abs_difference,
            "corr": corr
        },
        coords=model_da.coords,
        attrs={
            "description": "Per-gridpoint comparison metrics (Dask-based)",
            "note": "difference = model - obs, corr is time-wise correlation"
        }
    )
    return metrics_ds


def visualize_maps_cartopy(model_da, obs_da, metrics_ds, time_idx=0):
    """
    Cartopy-based map plots for a chosen time index.
    We'll trigger .compute() under the hood for the slices we plot.
    """
    # We assume lat-lon data in PlateCarree
    proj = ccrs.PlateCarree()

    # Force compute of the requested slices (small memory usage if time_idx is single)
    model_slice = model_da.isel(time=time_idx).compute()
    obs_slice   = obs_da.isel(time=time_idx).compute()
    diff_slice  = metrics_ds["difference"].isel(time=time_idx).compute()

    # 1) Model, Obs, Difference
    fig = plt.figure(figsize=(20, 6))

    ax1 = fig.add_subplot(1, 3, 1, projection=proj)
    ax1.set_title(f"Model (time={time_idx})")
    ax1.coastlines()
    ax1.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='gray')
    model_slice.plot(
        ax=ax1, transform=proj,
        x='lon', y='lat', cmap='viridis',
        cbar_kwargs={"label": str(model_da.name)}
    )

    ax2 = fig.add_subplot(1, 3, 2, projection=proj)
    ax2.set_title(f"Obs (time={time_idx})")
    ax2.coastlines()
    ax2.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='gray')
    obs_slice.plot(
        ax=ax2, transform=proj,
        x='lon', y='lat', cmap='viridis',
        cbar_kwargs={"label": str(obs_da.name)}
    )

    ax3 = fig.add_subplot(1, 3, 3, projection=proj)
    ax3.set_title(f"Difference (time={time_idx})")
    ax3.coastlines()
    ax3.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='gray')
    diff_slice.plot(
        ax=ax3, transform=proj,
        x='lon', y='lat', cmap='RdBu', center=0,
        cbar_kwargs={"label": "Difference"}
    )

    plt.tight_layout()
    plt.show()

    # 2) Correlation map (2D)
    corr2d = metrics_ds["corr"].compute()
    plt.figure(figsize=(8,6))
    ax4 = plt.subplot(1,1,1, projection=proj)
    ax4.set_title("Time-wise Correlation (Model vs. Obs)")
    ax4.coastlines()
    ax4.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='gray')
    corr2d.plot(
        ax=ax4, transform=proj,
        x='lon', y='lat', cmap='coolwarm', vmin=-1, vmax=1,
        cbar_kwargs={"label": "Corr"}
    )
    plt.tight_layout()
    plt.show()


def plot_global_metrics_bar(stats_dict):
    """
    Bar chart for extended metrics. We'll pick a subset to show.
    """
    # Decide which metrics to display
    metric_order = ["bias", "mae", "rmse", "corr", "std_ratio", "nse"]
    display = {k: stats_dict[k] for k in metric_order if k in stats_dict}

    labels = list(display.keys())
    values = [display[k] for k in labels]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, values, color='skyblue')
    plt.title("Extended Global Metrics")
    plt.ylabel("Metric Value")
    plt.xticks(rotation=45)

    for bar, val in zip(bars, values):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            height*1.01 if val >= 0 else height - 0.05,
            f"{val:.3f}",
            ha='center', va='bottom' if val >= 0 else 'top'
        )

    plt.tight_layout()
    plt.show()


############################################################
#                   MAIN SCRIPT
############################################################

def main():
    print("=== Model-vs-Observation Comparison (Dask, GPU optional) ===\n")
    
    # 1. Prompt for GPU usage
    use_gpu_input = input("Use GPU acceleration? (y/n) [default=n]: ").strip().lower()
    use_gpu = (use_gpu_input == 'y')
    if use_gpu and not GPU_AVAILABLE:
        print("CuPy not found or no GPU available; continuing on CPU.")
        use_gpu = False

    # 2. Setup Dask cluster
    workers_input = input("Enter number of workers [default=1]: ").strip()
    try:
        n_workers = int(workers_input) if workers_input else 1
    except:
        n_workers = 1
    client = init_dask_cluster(use_gpu=use_gpu, n_workers=n_workers)

    # 3. Paths
    model_path = input("Path to MODEL dataset: ")
    obs_path   = input("Path to OBS dataset: ")
    if not os.path.isfile(model_path):
        sys.exit(f"Model file not found: {model_path}")
    if not os.path.isfile(obs_path):
        sys.exit(f"Obs file not found: {obs_path}")

    # 4. Open with chunking
    #   We choose chunk sizes that are smaller to manage huge data.
    chunk_choice = input("Enter chunk size in the form 'time,lat,lon' [default=10,200,200]: ")
    if chunk_choice.strip():
        try:
            tchunk, lchunk, lonchunk = chunk_choice.split(',')
            chunks = {'time': int(tchunk), 'lat': int(lchunk), 'lon': int(lonchunk)}
        except:
            print("Invalid chunk input, using default.")
            chunks = None
    else:
        chunks = None
    
    # Open datasets
    model_ds = open_dataset_with_chunks(model_path, chunks=chunks, use_gpu=use_gpu)
    obs_ds   = open_dataset_with_chunks(obs_path, chunks=chunks, use_gpu=use_gpu)

    # 5. List variables in model & pick one
    model_vars = list(model_ds.data_vars.keys())
    if not model_vars:
        sys.exit("No variables in model dataset; cannot proceed.")
    print("\nVariables in model dataset:")
    for idx, var in enumerate(model_vars):
        print(f"[{idx}] {var}")
    var_choice = input("\nChoose variable by index or name: ")
    try:
        if var_choice.isdigit():
            variable_name = model_vars[int(var_choice)]
        else:
            variable_name = var_choice
        validate_variable_existence(model_ds, variable_name)
        validate_variable_existence(obs_ds, variable_name)
    except (ValueError, KeyError, IndexError) as e:
        sys.exit(f"Invalid variable choice: {e}")
    
    # 6. Detect dimension mismatch & optional regrid
    try:
        model_ds, obs_ds = detect_or_fix_mismatch(model_ds, obs_ds, variable_name)
    except (ValueError, KeyError) as e:
        sys.exit(f"Error fixing mismatch: {e}")

    # 7. NaN checks
    if check_missing_values(model_ds, variable_name):
        print("Warning: Model has NaNs.")
    if check_missing_values(obs_ds, variable_name):
        print("Warning: Obs has NaNs.")

    # 8. Extended global metrics
    model_da = model_ds[variable_name]
    obs_da   = obs_ds[variable_name]
    global_stats = extended_global_stats(model_da, obs_da)
    print("\n=== Extended Global Statistics ===")
    for k, v in global_stats.items():
        print(f"{k}: {v:.4f}")
    plot_global_metrics_bar(global_stats)

    # 9. Per-gridpoint metrics
    print("\nComputing gridwise metrics (difference, abs_difference, correlation). This may be large...")
    metrics_ds = compute_gridwise_metrics(model_da, obs_da)

    # We'll eventually save this, which triggers a compute of all results
    output_file = "comparison_metrics.nc"
    print(f"Saving metrics to {output_file} (this may take a while for large data)...")
    # to_netcdf with Dask: can handle big data. 
    # If you have NetCDF4 library with parallel I/O or zarr format, you can do more advanced parallel writes.
    metrics_ds.to_netcdf(output_file)
    print("Saved gridwise metrics.\n")

    # 10. Visualization
    time_idx_input = input("Enter a time index for the map plots [default=0]: ")
    if not time_idx_input.strip():
        time_idx = 0
    else:
        try:
            time_idx = int(time_idx_input)
        except:
            time_idx = 0
    print("Plotting maps with cartopy...")
    visualize_maps_cartopy(model_da, obs_da, metrics_ds, time_idx)

    print("\nDone! You can monitor resource usage at the Dask dashboard for large computations.")
    client.close()


if __name__ == "__main__":
    main()
