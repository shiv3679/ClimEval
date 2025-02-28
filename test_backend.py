import pytest
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from backend import (
    init_dask_cluster, open_dataset_with_chunks, detect_or_fix_mismatch,
    compute_continuous_metrics, compute_gridwise_continuous, visualize_map, visualize_metric_map
)

# Sample paths (update with your actual dataset paths)
MODEL_PATH = "../../Datasets/india_model_output_005deg_irregular_storm.nc"
OBS_PATH = "../../Datasets/india_obs_output_005deg_irregular_storm.nc"

def test_dask_cluster():
    """ Test if Dask cluster initializes properly """
    client = init_dask_cluster(use_gpu=False, n_workers=2)
    assert client is not None
    client.close()

def test_dataset_loading():
    """ Test if the NetCDF datasets load correctly with proper dimensions """
    model_ds = open_dataset_with_chunks(MODEL_PATH)
    obs_ds = open_dataset_with_chunks(OBS_PATH)
    assert "time" in model_ds.dims
    assert "lat" in model_ds.dims
    assert "lon" in model_ds.dims

def test_metrics_computation():
    """ Test computation of continuous metrics """
    model_ds = open_dataset_with_chunks(MODEL_PATH)
    obs_ds = open_dataset_with_chunks(OBS_PATH)

    variable = "lightning_density"  # Change this based on your dataset
    model_da = model_ds[variable]
    obs_da = obs_ds[variable]

    results = compute_continuous_metrics(model_da, obs_da)
    assert isinstance(results, dict)
    assert "MSE" in results
    assert "MAE" in results
    assert "Bias" in results
    assert "Corr" in results

def test_visualization():
    """ Test basic visualization of model vs observation """
    model_ds = open_dataset_with_chunks(MODEL_PATH)
    obs_ds = open_dataset_with_chunks(OBS_PATH)
    
    variable = "lightning_density"
    model_da = model_ds[variable]
    obs_da = obs_ds[variable]

    visualize_map(model_da, obs_da, time_index=0)  # Should display a plot without errors

def test_mae_rmse_visualization():
    """ Compute gridwise continuous metrics and plot MAE & RMSE maps """
    model_ds = open_dataset_with_chunks(MODEL_PATH)
    obs_ds = open_dataset_with_chunks(OBS_PATH)

    variable = "lightning_density"
    model_da = model_ds[variable]
    obs_da = obs_ds[variable]

    print("Computing Gridwise Continuous Metrics (Bias, MAE, MSE, Corr)")
    grid_metrics = compute_gridwise_continuous(model_da, obs_da).compute()

    # Ensure the expected keys exist
    assert "mae_map" in grid_metrics
    assert "mse_map" in grid_metrics

    # Plot MAE Map
    print("Displaying MAE Map")
    visualize_metric_map(grid_metrics["mae_map"], title_str="MAE Map", cmap="Reds")

    # Convert MSE to RMSE and visualize
    rmse_map = np.sqrt(grid_metrics["mse_map"])
    rmse_map.name = "rmse_map"
    print("Displaying RMSE Map")
    visualize_metric_map(rmse_map, title_str="RMSE Map", cmap="Purples")

if __name__ == "__main__":
    pytest.main(["-v"])
