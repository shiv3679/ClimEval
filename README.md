# EvalMetrics: Precision in Prediction


## OVERVIEW
This tool, "Model vs. Observation Verification (Per-Pixel Metrics)", is a Python-based script that:

1. Loads two NetCDF datasets (model vs. observation).
2. Handles large data with Dask for out-of-core computation.
3. Allows optional GPU usage (if CuPy and a compatible CUDA environment are available).
4. Checks for dimension mismatches and can regrid one dataset to the other's grid.
5. Computes a variety of domain-wide (aggregate) metrics:
    - Continuous metrics (MSE, MAE, Bias, Correlation)
    - Event-based metrics (POD, FAR, CSI, ETS, FSS)
    - Probabilistic metrics (Brier Score, BSS, etc.)
    - Distribution-level metrics (Wasserstein distance, Jensen-Shannon divergence)
6. Computes and visualizes "per-pixel" (gridwise) metrics across time for both continuous (Bias, MAE, MSE, Correlation maps) and event-based (POD, FAR, CSI, ETS maps).
7. Produces side-by-side plots for model vs. observation fields and also metric maps using cartopy for geographic context.
8. Optionally saves the results to a NetCDF file.


You can run this script interactively in a terminal and receive textual output plus pop-up map plots for each chosen metric.

## REQUIREMENTS
1. Python 3.7+ recommended

2. An environment with:

    - numpy
    - xarray
    - dask
    - distributed
    - matplotlib
    - cartopy
    - netCDF4
    - scipy

3. Optional for GPU usage:

    - cupy (plus a CUDA-capable NVIDIA GPU with compatible drivers)


Depending on your system, you may install via: 

```bash
pip install xarray dask distributed cupy netCDF4 scipy cartopy
```

Or use a conda environment: 

```bash
conda install xarray dask distributed netcdf4 scipy cartopy 
```

For GPU usage, you'll need to install CuPy and a compatible CUDA environment.




## SCRIPT USAGE


1. Launch your Python environment (conda, virtualenv, etc.).
2. Run: python final_with_vis.py
3. Follow the prompts: 
    - a) "Use GPU acceleration? (y/n)": If you have cupy + CUDA installed, type 'y' to use GPU. Otherwise, default to 'n'. 
    - b) "Number of Dask workers": For local parallelism. If you have multiple CPU cores or GPUs, you can specify more workers. 
    - c) "Path to MODEL dataset": Enter the path to the model NetCDF file.
    - d) "Path to OBS dataset": Enter the path to the observation NetCDF file. 
    - e) "Chunk sizes 'time,lat,lon' [default=10,200,200]": If you have very large data or a specific preference for chunking, specify them. Otherwise hit enter to use default.
    - f) The script checks if the dimensions match. If not, it will prompt you to regrid one dataset to the other. 
    - g) It lists the available variables in the model dataset. Select by index or by exact name (the same variable must exist in the obs dataset). 
    - h) (Optional) You can visualize any time index side-by-side for model vs obs. 
    - i) Select which domain-wide metrics to compute:
        1.  for continuous metrics
        2. for event-based metrics
        3. for probabilistic metrics
        4. for distribution-level metrics 
    - j) For event-based, specify a threshold. For example, '1.0' or '0.1'. If the data do not exceed that threshold, you'll see NaNs for hits/misses.
    - k) For probabilistic metrics, confirm if the model data are indeed probabilities (0..1). If yes, also specify the threshold to convert obs to 0/1. Otherwise skip.
    - l) Finally, decide whether you want per-pixel (gridwise) metrics. If 'y', the script computes and plots bias_map, corr_map, event-based POD/FAR maps, etc. 
    - m) The script will show a series of cartopy-based pop-up windows or inline plots (depending on your environment) for each chosen metric map.


## INTERPRETING THE OUTPUT

1. Domain-wide metrics: Printed in the console. If any computed value is NaN, it indicates some dimension of zero events or no overlapping data.

2. Per-pixel metrics:

    - "Bias map" shows how much the model over- or underestimates at each lat-lon, averaged over time.
    - "Corr map" reveals correlation over time for each lat-lon cell.
    - "POD map" shows Probability of Detection for threshold-based events per pixel.
    - "FAR map" is the False Alarm Ratio, etc. If the entire domain or particular grid cells have no events above the threshold, metrics can be NaN.
3. Warnings like "RuntimeWarning: invalid value encountered in divide" typically occur when a denominator is zero (leading to NaN). It's normal if the data never exceed your threshold or if there's no temporal variation for some grid cells.

4. If using Dask, you can monitor the Dask dashboard link (e.g., http://127.0.0.1:8787) to see tasks, memory usage, and parallel processing.

## COMMON ISSUES

- "NaN" event-based metrics: Usually means the threshold is not exceeded, or there's no overlap in events.
- "invalid value encountered in divide" warnings: Happens when the script calculates metrics that lead to dividing by zero. The final result for those cells is NaN.
- GPU not recognized: Ensure cupy is installed and your CUDA environment is set up. Otherwise, select CPU usage.


## EXTENSIONS & CUSTOMIZATIONS


- Add or remove metrics in the relevant compute functions.
- Implement advanced regridding (like xesmf) if linear interpolation is not sufficient.
- Adjust chunk sizes for performance or memory constraints.
- Save per-pixel metrics as NetCDF for offline or advanced analysis, e.g.: grid_cont.to_netcdf("gridwise_continuous_metrics.nc") grid_evt.to_netcdf("gridwise_event_metrics.nc")
- For real-time usage or a GUI, consider wrapping in a web framework (e.g., Panel, Streamlit).

## CREDITS & LICENSE

- Built with xarray, dask, cartopy, netCDF4, cupy, and other open-source Python libraries.
- You may distribute or modify this script as needed. No specific license text is provided here; adapt to your organizational requirements.