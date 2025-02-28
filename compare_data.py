#!/usr/bin/env python3

import xarray as xr
import numpy as np
import math
import matplotlib.pyplot as plt
import os
import sys

def validate_variable_existence(ds, var_name):
    """
    Checks if the specified variable exists in the dataset. 
    Raises KeyError if not found.
    """
    if var_name not in ds.data_vars:
        raise KeyError(f"Variable '{var_name}' not found in dataset. "
                       f"Available variables: {list(ds.data_vars.keys())}")

def detect_or_fix_mismatch(model_ds, obs_ds, var_name):
    """
    Checks if the model and obs datasets share the same dimensions for
    lat, lon, and time AND have matching coordinate arrays.

    If there is a mismatch, prompts the user to decide whether to:
      - Regrid model to obs,
      - Regrid obs to model,
      - or Exit.

    Returns updated (model_ds, obs_ds) that are guaranteed to match
    in (time, lat, lon) and coordinate values.
    """
    # Quick dimension presence check
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
    
    # If shapes match, we still want to check coordinate arrays
    # e.g., lat/lon/time might not be identical but have same length.
    # We'll do more thorough checks below.
    if same_shape:
        # Check coordinate arrays closely
        coords_match = True
        
        # For lat/lon, we'll allow "almost" the same via np.allclose
        # For time, we'll check exact equality. (Or you can decide to relax it.)
        if not np.allclose(model_ds['lat'].values, obs_ds['lat'].values):
            coords_match = False
        if not np.allclose(model_ds['lon'].values, obs_ds['lon'].values):
            coords_match = False
        if not np.array_equal(model_ds['time'].values, obs_ds['time'].values):
            coords_match = False
        
        if coords_match:
            # Perfect match. No regridding needed.
            print("Dataset dimensions and coordinates match—no regridding needed.\n")
            return model_ds, obs_ds
        else:
            print("Dataset dimensions match in size but coordinate values differ.")
    else:
        print("Mismatch in dimension sizes for (time, lat, or lon).")
    
    # If we get here, there's some kind of mismatch in shape or coordinate values.
    print("Detected mismatch in shape or coordinate arrays.")
    user_choice = input(
        "Would you like to (1) regrid one dataset to the other or (2) exit?\n"
        "Enter '1' to regrid, '2' to exit: "
    ).strip()
    
    if user_choice == '2':
        sys.exit("Exiting due to mismatch. Please reformat or align your data manually.")
    elif user_choice != '1':
        sys.exit("Invalid selection. Exiting.")
    
    # If user wants to regrid, ask which direction
    regrid_dir = input(
        "Enter 'm2o' to regrid Model -> Obs grid, or 'o2m' to regrid Obs -> Model grid: "
    ).strip()
    
    # We'll do a simple interpolation approach using xarray.interp_like
    # which will regrid lat/lon/time from one DS to the other's coordinate arrays.
    # If you need more sophisticated regridding (conservative, bilinear, etc.),
    # consider using xesmf or advanced methods.

    if regrid_dir.lower() == 'm2o':
        print("Regridding Model -> Observation coordinates ...")
        # We also need to ensure the variable of interest is present during interpolation
        # but usually xarray handles it well if it's in the dataset.
        model_ds = model_ds.interp_like(obs_ds, method='linear')
    elif regrid_dir.lower() == 'o2m':
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

def global_stats(model_data, obs_data):
    """
    Compute global (domain-wide) stats as single numbers:
      - bias
      - mae
      - rmse
      - correlation (over entire 1D flattened array)
    Returns a dictionary with these single-value metrics.
    """
    model_arr = model_data.values.flatten()
    obs_arr   = obs_data.values.flatten()
    
    bias = np.mean(model_arr - obs_arr)
    mae  = np.mean(np.abs(model_arr - obs_arr))
    rmse = math.sqrt(np.mean((model_arr - obs_arr)**2))
    
    corr_matrix = np.corrcoef(model_arr, obs_arr)
    corr_value  = corr_matrix[0, 1]
    
    return {
        'bias' : bias,
        'mae'  : mae,
        'rmse' : rmse,
        'corr' : corr_value
    }

def compute_gridwise_metrics(model_data, obs_data):
    """
    Compute per-gridpoint difference and absolute difference:
      - difference = model - obs (time, lat, lon)
      - abs_difference = |model - obs| (time, lat, lon)
    Also compute correlation across the time dimension for each lat-lon:
      - correlation => shape: (lat, lon)
    """
    # difference across (time, lat, lon)
    difference = model_data - obs_data
    abs_difference = np.abs(difference)
    
    # correlation across time dimension for each lat-lon
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
      1) Model field at a given time step
      2) Observation field at the same time step
      3) (Model - Obs) difference for that time step
      4) Correlation map over lat-lon (computed across the entire time dimension)
    """
    plt.figure(figsize=(18, 6))
    
    # Subplot 1: Model
    ax1 = plt.subplot(1, 3, 1)
    model_slice = model_data.isel(time=time_step)
    im1 = model_slice.plot(ax=ax1, x='lon', y='lat', cmap='viridis')
    ax1.set_title(f"Model (time={time_step})")
    plt.colorbar(im1, ax=ax1, orientation='vertical', label=str(model_data.name))
    
    # Subplot 2: Observations
    ax2 = plt.subplot(1, 3, 2)
    obs_slice = obs_data.isel(time=time_step)
    im2 = obs_slice.plot(ax=ax2, x='lon', y='lat', cmap='viridis')
    ax2.set_title(f"Obs (time={time_step})")
    plt.colorbar(im2, ax=ax2, orientation='vertical', label=str(obs_data.name))
    
    # Subplot 3: Difference (Model - Obs)
    ax3 = plt.subplot(1, 3, 3)
    diff_slice = metrics_ds['difference'].isel(time=time_step)
    im3 = diff_slice.plot(ax=ax3, x='lon', y='lat', cmap='RdBu', center=0)
    ax3.set_title(f"Difference (time={time_step})")
    plt.colorbar(im3, ax=ax3, orientation='vertical', label='Difference')
    
    plt.tight_layout()
    plt.show()
    
    # Correlation (lat-lon)
    plt.figure(figsize=(8, 6))
    corr = metrics_ds['corr']
    im4 = corr.plot(x='lon', y='lat', cmap='coolwarm', vmin=-1, vmax=1)
    plt.title("Time-wise Correlation (Model vs. Obs)")
    plt.colorbar(im4, orientation='vertical', label='Correlation Coeff.')
    plt.tight_layout()
    plt.show()

def main():
    """
    Main function that:
     1) Prompts the user for file paths.
     2) Opens the datasets.
     3) Lists variables in the model dataset so user can pick one.
     4) Checks for mismatches in lat/lon/time and offers regridding.
     5) Computes metrics, saves to NetCDF.
     6) Visualizes results.
    """
    print("Welcome to the Model-vs-Observation Comparison Tool (with regridding)!\n")
    
    # 1. Get user inputs (files)
    model_path = input("Enter the path to the MODEL dataset (e.g., /path/to/model.nc): ")
    obs_path   = input("Enter the path to the OBSERVATION dataset (e.g., /path/to/obs.nc): ")
    
    # Quick checks for file existence
    if not os.path.isfile(model_path):
        sys.exit(f"Error: Model file not found at '{model_path}'")
    if not os.path.isfile(obs_path):
        sys.exit(f"Error: Observation file not found at '{obs_path}'")
    
    # 2. Load the data
    try:
        model_ds = xr.open_dataset(model_path)
        obs_ds   = xr.open_dataset(obs_path)
    except Exception as e:
        sys.exit(f"Error loading datasets: {e}")
    
    # 3. Show variables in the model dataset and ask user to pick
    model_vars = list(model_ds.data_vars.keys())
    if not model_vars:
        sys.exit("No variables found in the model dataset. Cannot proceed.")
    
    print("\nAvailable variables in MODEL dataset:")
    for idx, var in enumerate(model_vars):
        print(f"  [{idx}] {var}")
    
    var_choice = input("\nEnter the index or exact name of the variable you want to compare: ")
    try:
        if var_choice.isdigit():
            var_index = int(var_choice)
            variable_name = model_vars[var_index]
        else:
            # user typed the exact variable name
            if var_choice not in model_vars:
                raise ValueError(f"'{var_choice}' not found among model dataset variables.")
            variable_name = var_choice
    except (ValueError, IndexError) as e:
        sys.exit(f"Invalid selection: {e}")
    
    print(f"\nYou selected variable: {variable_name}")
    
    # Confirm the variable also exists in the obs dataset
    if variable_name not in obs_ds.data_vars:
        print(f"Obs dataset variables: {list(obs_ds.data_vars.keys())}")
        sys.exit(f"Error: Variable '{variable_name}' does not exist in the observation dataset.")
    
    # 4. Check for dimension mismatch & possibly regrid
    print("\nChecking for dimension and coordinate mismatches...\n")
    try:
        model_ds, obs_ds = detect_or_fix_mismatch(model_ds, obs_ds, variable_name)
    except (ValueError, KeyError) as e:
        sys.exit(f"Error while checking/fixing mismatch: {e}")
    
    # 5. Check for NaNs
    if check_missing_values(model_ds, variable_name):
        print("Warning: Model data contains NaN values.")
    if check_missing_values(obs_ds, variable_name):
        print("Warning: Observations contain NaN values.")
    
    # 6. Prompt user for time index to visualize
    time_index_input = input("\nEnter the TIME INDEX to visualize (0-based). Press ENTER for default (0): ")
    if time_index_input.strip() == "":
        time_index = 0
    else:
        try:
            time_index = int(time_index_input)
        except:
            print("Invalid time index. Using 0.")
            time_index = 0
    
    # 7. Extract DataArrays after potential regridding
    model_da = model_ds[variable_name]
    obs_da   = obs_ds[variable_name]
    
    # 8. Compute global statistics
    stats = global_stats(model_da, obs_da)
    print("\n=== Global (Domain-Wide) Statistics ===")
    print(f"  Bias (model - obs): {stats['bias']:.4f}")
    print(f"  MAE               : {stats['mae']:.4f}")
    print(f"  RMSE              : {stats['rmse']:.4f}")
    print(f"  Correlation       : {stats['corr']:.4f}\n")
    
    # 9. Compute gridwise metrics
    print("Computing gridwise metrics (difference, abs_difference, correlation)...")
    metrics_ds = compute_gridwise_metrics(model_da, obs_da)
    
    # 10. Save results
    output_name = "comparison_metrics.nc"
    metrics_ds.to_netcdf(output_name)
    print(f"Saved gridwise comparison metrics to '{output_name}'\n")
    
    # 11. Visualization
    print("Generating visualization for the user-selected time step...")
    try:
        visualize_comparison(model_da, obs_da, metrics_ds, time_step=time_index)
    except IndexError:
        print("Warning: The time index provided is out of range. Skipping visualization.")
    except Exception as e:
        print(f"Error during visualization: {e}")

    print("Comparison Tool has finished. Goodbye!")

if __name__ == "__main__":
    main()
