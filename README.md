# 🌍 ClimEval: Climate Model vs. Observation Verification Tool




## 📌 OVERVIEW

ClimEval is a Python package designed for **climate model comparisons** against observation datasets. It enables researchers and analysts to:



- Efficiently compare **climate model output vs. observational data**.
- Compute a range of **evaluation metrics** (continuous, event-based, probabilistic, and distribution-based).
- Perform **gridwise per-pixel analysis** to assess spatial biases and detection capabilities.
- Use **Dask** for parallel computing and **GPU acceleration** (optional).
- Generate **cartopy-based visualizations** of climate fields and statistical maps


## 🔧 Installation
🛠 Install via PyPI:
```bash
pip install climeval
```

🐍 Install from Source (Development Version):

```bash

git clone https://github.com/shiv3679/ClimEval.git
cd ClimEval
pip install .
```

## 🖥 Dependencies:
ClimEval requires Python 3.7+ and the following dependencies:

- `numpy`, `xarray`, `dask`, `scipy`
- `matplotlib`, `cartopy`, `pytest`
- Optional for GPU: `cupy` (requires CUDA)


To install dependencies separately:

```bash
pip install numpy xarray dask scipy matplotlib cartopy pytest
```

For GPU acceleration (optional, requires CUDA):

```bash
pip install cupy
```

## Usage Guide

ClimEval can be used via CLI (Command-Line Interface) or Python API.

### CLI Usage

```bash
climeval --help
```

Example Command:

```bash
climeval india_model_output_005deg_irregular_storm.nc india_obs_output_005deg_irregular_storm.nc --variable lightning_density --threshold 1.0 --visualize
```
### CLI Options:

```bash
Usage: climeval [OPTIONS] MODEL_PATH OBS_PATH

Options:
  --variable TEXT       Variable to evaluate (e.g., 'temperature')
  --chunks TEXT         Dask chunking format (default: {'time': 10, 'lat': 200, 'lon': 200})
  --threshold FLOAT     Threshold for event-based metrics (default: 1.0)
  --visualize           Generate visualizations
  -h, --help           Show this message and exit.

```

### Python API Usage

You can use `ClimEval` directly in Python scripts:


**Example Usage**

```python
import xarray as xr
from climeval.core import compute_continuous_metrics, compute_event_metrics
from climeval.visualization import visualize_map, visualize_metric_map

# Load Datasets
model_ds = xr.open_dataset("model.nc")
obs_ds = xr.open_dataset("obs.nc")

# Select Variable
variable = "lightning_density"
model_da = model_ds[variable]
obs_da = obs_ds[variable]

# Compute Metrics
cont_metrics = compute_continuous_metrics(model_da, obs_da)
event_metrics = compute_event_metrics(model_da, obs_da, threshold=1.0)

print("Continuous Metrics:", cont_metrics)
print("Event-Based Metrics:", event_metrics)

# Visualization
visualize_map(model_da, obs_da, time_index=0)

```


## 📊 Metrics Computed
ClimEval calculates multiple climate verification metrics, categorized as:

 1. Continuous Metrics (Domain-wide)

    - MSE (Mean Squared Error)
    - MAE (Mean Absolute Error)
    - Bias (Mean Difference)
    - Correlation Coefficient (Corr)
2.  Event-Based Metrics

    - POD (Probability of Detection)
    - FAR (False Alarm Ratio)
    - CSI (Critical Success Index)
    - ETS (Equitable Threat Score)
    - FSS (Fractions Skill Score)

3. Probabilistic Metrics

    - Brier Score (BS)
    - Brier Skill Score (BSS)
    - Ranked Probability Skill Score (RPSS)

4. Distribution-Based Metrics

    - Wasserstein Distance
    - Jensen-Shannon Divergence

5.  Per-Pixel (Gridwise) Metrics

    - Bias Map
    - MAE Map
    - MSE Map
    - Correlation Map
    - POD, FAR, CSI, and ETS maps


## 🗺 Visualizations
ClimEval provides cartopy-based visualizations for model-observation comparison.

Example **side-by-side model vs. observation plot**:

```python
from climeval.visualization import visualize_map
visualize_map(model_da, obs_da, time_index=0)
```

Example **BIAS Map** Visualization

```python
from climeval.visualization import visualize_metric_map
bias_map = model_da.mean(dim="time") - obs_da.mean(dim="time")
visualize_metric_map(bias_map, title_str="Bias Map", cmap="RdBu", center=0)
```

## ⚠️ Common Issues & Fixes
### NaN in Event-Based Metrics?

- Try reducing `--threshold` (e.g., `--threshold 0.5`).
- Ensure the dataset contains meaningful events above the threshold.
### Wayland Warning on GNOME?

- Fix by running:
```bash
export QT_QPA_PLATFORM=wayland
```
- Alternative:
```bash
export QT_QPA_PLATFORM=xcb
```


### Dask Errors?

- Ensure no other Dask clusters are running (port 8787 conflicts).
- Run:
```bash
pkill -f dask
```
to close any existing Dask processes.


### GPU Not Recognized?

- Ensure cupy is installed and CUDA drivers are configured.
- Run:
```python
import cupy as cp
print(cp.cuda.Device(0))
```
to check GPU availability.


## 🏗 Future Enhancements
- ✔️ **Better GPU Integration** with optimized Dask-CuPy workflows.
- ✔️ **Integration with AI-Based Climate Predictions**.
- ✔️ **More Advanced Regridding Methods (xesmf support).**
- ✔️ **Interactive Web Interface for Metrics Visualization.**


## 📜 License

ClimEval is **open-source** under the **MIT License**.

Feel free to contribute, improve, or modify it.

Contributions are welcome!


## 👨‍💻 Author
- 📌 Developed by Shiv Shankar Singh
- 🌐 GitHub: shiv3679
- 📧 Email: shivshankarsingh.py@gmail.com



