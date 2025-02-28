import climeval
from climeval.utils import open_dataset_with_chunks, detect_or_fix_mismatch
from climeval.core import compute_continuous_metrics, compute_event_metrics
from climeval.visualization import visualize_map, visualize_metric_map
import os

# ✅ Set your dataset paths
MODEL_PATH = "../../Datasets/india_model_output_005deg_irregular_storm.nc"
OBS_PATH = "../../Datasets/india_obs_output_005deg_irregular_storm.nc"

def main():
    print("\n✅ Step 1: Loading datasets...")
    model_ds = open_dataset_with_chunks(MODEL_PATH, chunks={"time": 10, "lat": 200, "lon": 200})
    obs_ds = open_dataset_with_chunks(OBS_PATH, chunks={"time": 10, "lat": 200, "lon": 200})

    assert "time" in model_ds.dims, "Error: 'time' dimension missing in model dataset"
    assert "lat" in model_ds.dims, "Error: 'lat' dimension missing in model dataset"
    assert "lon" in model_ds.dims, "Error: 'lon' dimension missing in model dataset"
    print("✅ Datasets loaded successfully")

    # ✅ Step 2: Check for Mismatches
    print("\n✅ Step 2: Checking for mismatches...")
    model_ds, obs_ds = detect_or_fix_mismatch(model_ds, obs_ds)
    print("✅ Mismatch check completed")

    # ✅ Step 3: Compute Metrics
    variable = "lightning_density"
    model_da = model_ds[variable]
    obs_da = obs_ds[variable]

    print("\n✅ Step 3: Computing continuous metrics...")
    cont_metrics = compute_continuous_metrics(model_da, obs_da)
    print("✅ Continuous Metrics:", cont_metrics)

    print("\n✅ Step 4: Computing event-based metrics...")
    event_metrics = compute_event_metrics(model_da, obs_da, threshold=1.0)
    print("✅ Event-Based Metrics:", event_metrics)

    # ✅ Step 5: Quick Visualizations
    print("\n✅ Step 5: Displaying sample visualizations...")
    visualize_map(model_da, obs_da, time_index=0)

    # ✅ Step 6: Gridwise Metrics Visualization
    print("\n✅ Step 6: Displaying bias map...")
    bias_map = model_da.mean(dim="time") - obs_da.mean(dim="time")
    visualize_metric_map(bias_map, title_str="Bias Map", cmap="RdBu", center=0)

if __name__ == "__main__":
    main()
