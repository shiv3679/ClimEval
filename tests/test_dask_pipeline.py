import climeval
from climeval.dask_setup import init_dask_cluster
from climeval.utils import open_dataset_with_chunks, detect_or_fix_mismatch
from climeval.core import compute_continuous_metrics, compute_event_metrics
from climeval.visualization import visualize_map, visualize_metric_map
import os
import dask.array as da

# ✅ Set dataset paths
MODEL_PATH = "../../Datasets/india_model_output_005deg_irregular_storm.nc"
OBS_PATH = "../../Datasets/india_obs_output_005deg_irregular_storm.nc"

def main():
    print("\n✅ Step 1: Initializing Dask cluster...")
    client = init_dask_cluster(use_gpu=False, n_workers=2)
    print(f"Dask dashboard available at: {client.dashboard_link}\n")

    # ✅ Step 2: Loading datasets using Dask
    print("\n✅ Step 2: Loading datasets with Dask...")
    model_ds = open_dataset_with_chunks(MODEL_PATH, chunks={"time": 10, "lat": 200, "lon": 200})
    obs_ds = open_dataset_with_chunks(OBS_PATH, chunks={"time": 10, "lat": 200, "lon": 200})
    print("✅ Datasets loaded successfully with Dask!")

    # ✅ Step 3: Checking for Mismatches
    print("\n✅ Step 3: Checking for mismatches...")
    model_ds, obs_ds = detect_or_fix_mismatch(model_ds, obs_ds)
    print("✅ Mismatch check completed")

    # ✅ Step 4: Compute Metrics with Dask
    variable = "lightning_density"
    model_da = model_ds[variable]
    obs_da = obs_ds[variable]

    print("\n✅ Step 4: Computing continuous metrics with Dask...")
    cont_metrics = compute_continuous_metrics(model_da, obs_da)
    print("✅ Continuous Metrics:", cont_metrics)

    print("\n✅ Step 5: Computing event-based metrics with Dask...")
    event_metrics = compute_event_metrics(model_da, obs_da, threshold=1.0)
    print("✅ Event-Based Metrics:", event_metrics)

    # ✅ Step 6: Quick Visualizations
    print("\n✅ Step 6: Displaying sample visualizations...")
    visualize_map(model_da, obs_da, time_index=0)

    # ✅ Step 7: Gridwise Metrics Visualization using Dask
    print("\n✅ Step 7: Computing & Displaying Bias Map with Dask...")
    bias_map = (model_da - obs_da).mean(dim="time")
    bias_map = bias_map.compute()  # Convert to NumPy for plotting
    visualize_metric_map(bias_map, title_str="Bias Map (Dask)", cmap="RdBu", center=0)

    # ✅ Step 8: Clean Up Dask Client
    print("\n✅ Closing Dask client...")
    client.close()
    print("✅ Dask client closed successfully!")

if __name__ == "__main__":
    main()
