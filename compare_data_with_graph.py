#!/usr/bin/env python3

import xarray as xr
import numpy as np
import math
import matplotlib.pyplot as plt
import os
import sys

# For map projections
import cartopy.crs as ccrs
import cartopy.feature as cfeature

###########################################
#            HELPER FUNCTIONS
###########################################

def validate_variable_existence(ds, var_name):
    """
    Checks if the specified variable exists in the dataset. 
    Raises KeyError if not found.
    """
    if var_name not in ds.data_vars:
        raise KeyError(
            f"Variable '{var_name}' not found in dataset. "
            f"Available variables: {list(ds.data_vars.keys())}"
        )

def detect_or_fix_mismatch(model_ds, obs_ds, var_name):
    """
    Checks if the model and obs datasets share the same dimensions for
    lat, lon, and time AND have matching coordinate arrays.

    If there is a mismatch, the user can choose to:
      - regrid model to obs,
      - regrid obs to model,
      - or exit.

    Returns updated (model_ds, obs_ds) that match in (time, lat, lon).
    """
    required_dims = ['time', 'lat', 'lon']
    for dim in required_dims:
        if dim not in model_ds.dims:
            raise ValueError(f"Model dataset missing dimension: {dim}")
        if dim not in obs_ds.dims:
            raise ValueError(f"Obs dataset missing dimension: {dim}")
    
    # Check if shapes match
    same_shape = True
    for dim in required_dims:
        if model_ds.dims[dim] != obs_ds.dims[dim]:
            same_shape = False
            break
    
    if same_shape:
        # Check coordinate arrays in detail
        coords_match = True
        # For lat/lon: allow near equality, for time: require exact
        if not np.allclose(model_ds['lat'].values, obs_ds['lat'].values):
            coords_match = False
        if not np.allclose(model_ds['lon'].values, obs_ds['lon'].values):
            coords_match = False
        if not np.array_equal(model_ds['time'].values, obs_ds['time'].values):
            coords_match = False
        
        if coords_match:
            print("Dataset dimensions and coordinates match—no regridding needed.\n")
            return model_ds, obs_ds
        else:
            print("Dimension sizes match, but coordinate values differ.")
    else:
        print("Mismatch in dimension sizes for (time, lat, or lon).")
    
    # If we reach here, there's a mismatch
    print("Detected mismatch in shape or coordinate arrays.")
    user_choice = input(
        "Would you like to (1) regrid one dataset to the other, or (2) exit?\n"
        "Enter '1' to regrid, '2' to exit: "
    ).strip()
    
    if user_choice == '2':
        sys.exit("Exiting. Please reformat or align your data manually.")
    elif user_choice != '1':
        sys.exit("Invalid selection. Exiting.")
    
    # Ask which direction for regridding
    regrid_dir = input(
        "Enter 'm2o' to regrid Model -> Obs, or 'o2m' to regrid Obs -> Model: "
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
    Checks for missing values (NaNs) in the given variable of a Dataset.
    Returns True if NaNs are found, otherwise False.
    """
    data = ds[var_name].values
    return np.isnan(data).any()

def extended_global_stats(model_data, obs_data):
    """
    Compute extended domain-wide metrics:
      - bias
      - mae
      - rmse
      - correlation (Pearson)
      - std_model
      - std_obs
      - std_ratio = std_model / std_obs
      - NSE (Nash-Sutcliffe Efficiency)

    Returns a dictionary with these metrics.
    """
    model_arr = model_data.values.flatten()
    obs_arr   = obs_data.values.flatten()
    
    bias = np.mean(model_arr - obs_arr)
    mae  = np.mean(np.abs(model_arr - obs_arr))
    rmse = math.sqrt(np.mean((model_arr - obs_arr)**2))
    
    corr_matrix = np.corrcoef(model_arr, obs_arr)
    corr_value  = corr_matrix[0, 1]
    
    std_m = np.std(model_arr)
    std_o = np.std(obs_arr)
    std_ratio = std_m / std_o if std_o != 0 else np.nan
    
    # Nash-Sutcliffe Efficiency
    numerator   = np.sum((model_arr - obs_arr)**2)
    denominator = np.sum((obs_arr - obs_arr.mean())**2)
    nse = 1.0 - (numerator / denominator) if denominator != 0 else np.nan
    
    return {
        'bias': bias,
        'mae': mae,
        'rmse': rmse,
        'corr': corr_value,
        'std_model': std_m,
        'std_obs': std_o,
        'std_ratio': std_ratio,
        'nse': nse
    }

def compute_gridwise_metrics(model_data, obs_data):
    """
    Compute per-gridpoint (time, lat, lon) metrics:
      - difference = model - obs
      - abs_difference = |model - obs|
      - corr = correlation across time dimension => shape (lat, lon)

    Returns an xarray.Dataset with these fields.
    """
    difference = model_data - obs_data
    abs_difference = np.abs(difference)
    
    model_mean = model_data.mean(dim='time')
    obs_mean   = obs_data.mean(dim='time')
    
    model_anom = model_data - model_mean
    obs_anom   = obs_data - obs_mean
    
    cov = (model_anom * obs_anom).mean(dim='time')
    model_std = model_data.std(dim='time')
    obs_std   = obs_data.std(dim='time')
    
    corr = cov / (model_std * obs_std)
    
    metrics_ds = xr.Dataset(
        {
            "difference": difference,
            "abs_difference": abs_difference,
            "corr": corr
        },
        coords=model_data.coords,  # includes time, lat, lon
        attrs={
            "description": "Per-gridpoint comparison metrics",
            "note": "difference = model - obs, corr is time-wise correlation per lat-lon"
        }
    )
    return metrics_ds

def visualize_comparison(model_data, obs_data, metrics_ds, time_step=0):
    """
    Visualizes:
      1) Model field at a chosen time step (map with cartopy)
      2) Observations field at the same time step
      3) (Model - Obs) difference at that time step
      4) Correlation map (lat-lon), computed across time
    """
    # We use a PlateCarree projection for lat-lon data
    projection = ccrs.PlateCarree()
    
    # ---------- Figure 1: Model, Obs, and Difference ----------
    fig = plt.figure(figsize=(20, 6))

    # Subplot 1: Model
    ax1 = fig.add_subplot(1, 3, 1, projection=projection)
    ax1.set_title(f"Model (time={time_step})")
    ax1.coastlines()
    ax1.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='gray')
    model_slice = model_data.isel(time=time_step)
    # The 'transform' tells cartopy the data is in lat/lon coordinates
    im1 = model_slice.plot(
        ax=ax1, transform=ccrs.PlateCarree(),
        x='lon', y='lat', cmap='viridis',
        cbar_kwargs={"label": str(model_data.name)}
    )

    # Subplot 2: Observations
    ax2 = fig.add_subplot(1, 3, 2, projection=projection)
    ax2.set_title(f"Obs (time={time_step})")
    ax2.coastlines()
    ax2.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='gray')
    obs_slice = obs_data.isel(time=time_step)
    im2 = obs_slice.plot(
        ax=ax2, transform=ccrs.PlateCarree(),
        x='lon', y='lat', cmap='viridis',
        cbar_kwargs={"label": str(obs_data.name)}
    )

    # Subplot 3: Difference (Model - Obs)
    ax3 = fig.add_subplot(1, 3, 3, projection=projection)
    ax3.set_title(f"Difference (time={time_step})")
    ax3.coastlines()
    ax3.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='gray')
    diff_slice = metrics_ds['difference'].isel(time=time_step)
    im3 = diff_slice.plot(
        ax=ax3, transform=ccrs.PlateCarree(),
        x='lon', y='lat', cmap='RdBu', center=0,
        cbar_kwargs={"label": "Difference"}
    )

    plt.tight_layout()
    plt.show()

    # ---------- Figure 2: Correlation Map (2D) ----------
    plt.figure(figsize=(8, 6))
    corr = metrics_ds['corr']
    ax4 = plt.subplot(1, 1, 1, projection=projection)
    ax4.set_title("Time-wise Correlation (Model vs. Obs)")
    ax4.coastlines()
    ax4.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='gray')
    im4 = corr.plot(
        ax=ax4, transform=ccrs.PlateCarree(),
        x='lon', y='lat', cmap='coolwarm', vmin=-1, vmax=1,
        cbar_kwargs={"label": "Correlation"}
    )
    plt.tight_layout()
    plt.show()

def plot_global_metrics_bar(stats_dict):
    """
    Creates a bar chart of the extended global metrics (bias, mae, rmse, corr, std_ratio, nse, etc.).
    You can customize which metrics to include in the bar chart by reordering or filtering below.
    """
    # Decide which metrics to display in the bar chart (and in what order)
    # You can reorder or remove items as you wish
    metric_order = ["bias", "mae", "rmse", "corr", "std_ratio", "nse"]
    display_stats = {k: stats_dict[k] for k in metric_order if k in stats_dict}

    labels = list(display_stats.keys())
    values = [display_stats[k] for k in labels]
    
    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, values, color='skyblue')
    plt.title("Extended Global Metrics")
    plt.ylabel("Metric Value")
    plt.xticks(rotation=45)
    
    # Optionally annotate each bar with its value
    for bar, val in zip(bars, values):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2., 
            1.01*height if val >= 0 else height - 0.05, 
            f"{val:.3f}",
            ha='center', va='bottom' if val >= 0 else 'top',
            rotation=0
        )
    
    plt.tight_layout()
    plt.show()

###########################################
#               MAIN SCRIPT
###########################################

def main():
    """
    Main function that:
     1) Prompts the user for file paths.
     2) Opens the datasets.
     3) Lists variables in the model dataset so user can pick one.
     4) Checks for dimension mismatch & offers regridding.
     5) Computes extended global metrics (including std dev & NSE).
     6) Plots a bar chart of the global metrics.
     7) Computes and saves gridwise metrics to NetCDF.
     8) Visualizes data/maps with cartopy.
    """
    print("Welcome to the Model-vs-Observation Comparison Tool (with regridding + extra metrics)!\n")
    
    # 1. File paths
    model_path = input("Enter the path to the MODEL dataset (e.g., /path/to/model.nc): ")
    obs_path   = input("Enter the path to the OBSERVATION dataset (e.g., /path/to/obs.nc): ")
    
    if not os.path.isfile(model_path):
        sys.exit(f"Error: Model file not found at '{model_path}'")
    if not os.path.isfile(obs_path):
        sys.exit(f"Error: Observation file not found at '{obs_path}'")
    
    # 2. Load
    try:
        model_ds = xr.open_dataset(model_path)
        obs_ds   = xr.open_dataset(obs_path)
    except Exception as e:
        sys.exit(f"Error loading datasets: {e}")
    
    # 3. Show variables in the model dataset, ask user to pick
    model_vars = list(model_ds.data_vars.keys())
    if not model_vars:
        sys.exit("No variables found in the model dataset. Cannot proceed.")
    
    print("\nAvailable variables in the MODEL dataset:")
    for idx, var in enumerate(model_vars):
        print(f"  [{idx}] {var}")
    
    var_choice = input("\nEnter the index or exact name of the variable to compare: ")
    try:
        if var_choice.isdigit():
            var_index = int(var_choice)
            variable_name = model_vars[var_index]
        else:
            if var_choice not in model_vars:
                raise ValueError(f"'{var_choice}' not in model dataset variables.")
            variable_name = var_choice
    except (ValueError, IndexError) as e:
        sys.exit(f"Invalid variable selection: {e}")
    
    if variable_name not in obs_ds.data_vars:
        sys.exit(
            f"Error: Variable '{variable_name}' does not exist in the observation dataset.\n"
            f"Obs dataset variables: {list(obs_ds.data_vars.keys())}"
        )
    
    print(f"\nYou selected variable: {variable_name}\n")
    
    # 4. Check dimension mismatch & regrid if needed
    print("Checking for dimension/coordinate mismatches...\n")
    try:
        model_ds, obs_ds = detect_or_fix_mismatch(model_ds, obs_ds, variable_name)
    except (ValueError, KeyError) as e:
        sys.exit(f"Error while checking/fixing mismatch: {e}")
    
    # 5. Warn if NaNs
    if check_missing_values(model_ds, variable_name):
        print("Warning: Model data contains NaN values.")
    if check_missing_values(obs_ds, variable_name):
        print("Warning: Observation data contains NaN values.")
    
    # 6. Prompt user for time index to visualize
    time_index_input = input("\nEnter the TIME INDEX to visualize (0-based). Press ENTER for 0: ")
    if time_index_input.strip() == "":
        time_index = 0
    else:
        try:
            time_index = int(time_index_input)
        except:
            print("Invalid time index. Using 0.")
            time_index = 0
    
    # Extract DataArrays
    model_da = model_ds[variable_name]
    obs_da   = obs_ds[variable_name]
    
    # 7. Compute extended global statistics
    stats = extended_global_stats(model_da, obs_da)
    print("\n=== Extended Global (Domain-Wide) Statistics ===")
    print(f"  Bias (model - obs):  {stats['bias']:.4f}")
    print(f"  MAE                :  {stats['mae']:.4f}")
    print(f"  RMSE               :  {stats['rmse']:.4f}")
    print(f"  Correlation        :  {stats['corr']:.4f}")
    print(f"  Std(Model)         :  {stats['std_model']:.4f}")
    print(f"  Std(Obs)           :  {stats['std_obs']:.4f}")
    print(f"  Std Ratio          :  {stats['std_ratio']:.4f}")
    print(f"  Nash-Sutcliffe Eff :  {stats['nse']:.4f}")
    
    # 8. Bar chart of these global metrics
    plot_global_metrics_bar(stats)
    
    # 9. Compute gridwise metrics
    print("\nComputing gridwise metrics (difference, abs_difference, correlation) ...")
    metrics_ds = compute_gridwise_metrics(model_da, obs_da)
    
    # 10. Save the metrics to NetCDF
    output_name = "comparison_metrics.nc"
    metrics_ds.to_netcdf(output_name)
    print(f"Saved gridwise comparison metrics to '{output_name}'\n")
    
    # 11. Visualize with cartopy
    print("Generating visualizations with cartopy (may take a moment) ...")
    try:
        visualize_comparison(model_da, obs_da, metrics_ds, time_step=time_index)
    except IndexError:
        print("Warning: The time index provided is out of range. Skipping visualization.")
    except Exception as e:
        print(f"Error during visualization: {e}")
    
    print("\nComparison Tool has finished. Goodbye!")

if __name__ == "__main__":
    main()
