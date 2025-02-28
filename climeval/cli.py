#climeeval/cli.py 

import argparse
import xarray as xr
from climeval.utils import open_dataset_with_chunks, detect_or_fix_mismatch
from climeval.core import compute_continuous_metrics, compute_event_metrics
from climeval.visualization import visualize_map, visualize_metric_map

def main():
    parser = argparse.ArgumentParser(description="ClimEval: Climate Model Evaluation Tool")

    parser.add_argument("model_path", type=str, help="Path to the model NetCDF file")
    parser.add_argument("obs_path", type=str, help="Path to the observation NetCDF file")
    parser.add_argument("--variable", type=str, required=True, help="Variable to evaluate (e.g., 'temperature')")
    parser.add_argument("--chunks", type=str, default="{'time': 10, 'lat': 200, 'lon': 200}",
                        help="Dask chunking format (default: {'time': 10, 'lat': 200, 'lon': 200})")
    parser.add_argument("--threshold", type=float, default=1.0, help="Threshold for event-based metrics (default: 1.0)")
    parser.add_argument("--visualize", action="store_true", help="Generate visualizations")

    args = parser.parse_args()

    print("\n✅ Step 1: Loading datasets...")
    model_ds = open_dataset_with_chunks(args.model_path, eval(args.chunks))
    obs_ds = open_dataset_with_chunks(args.obs_path, eval(args.chunks))

    print("\n✅ Step 2: Checking for mismatches...")
    model_ds, obs_ds = detect_or_fix_mismatch(model_ds, obs_ds)

    print("\n✅ Step 3: Computing continuous metrics...")
    model_da = model_ds[args.variable]
    obs_da = obs_ds[args.variable]
    cont_metrics = compute_continuous_metrics(model_da, obs_da)
    print("✅ Continuous Metrics:", cont_metrics)

    print("\n✅ Step 4: Computing event-based metrics...")
    event_metrics = compute_event_metrics(model_da, obs_da, threshold=args.threshold)
    print("✅ Event-Based Metrics:", event_metrics)

    if args.visualize:
        print("\n✅ Step 5: Displaying sample visualizations...")
        visualize_map(model_da, obs_da, time_index=0)

        print("\n✅ Step 6: Displaying bias map...")
        bias_map = model_da.mean(dim="time") - obs_da.mean(dim="time")
        visualize_metric_map(bias_map, title_str="Bias Map", cmap="RdBu", center=0)

if __name__ == "__main__":
    main()

