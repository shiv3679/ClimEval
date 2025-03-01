"""
ClimEval: Climate Model vs. Observation Verification Tool
---------------------------------------------------------
ClimEval is a package designed to compare climate model outputs with observational datasets.
It provides domain-wide and per-grid validation metrics, visualization tools, and efficient
computation with Dask for large datasets.

Modules:
- `core`: Computes continuous, event-based, probabilistic, and distributional metrics.
- `visualization`: Contains functions for visualizing spatial maps and time-series comparisons.
- `utils`: Utility functions for loading datasets and handling dimension mismatches.
- `dask_setup`: Functions for initializing and managing Dask clusters.

Example usage:
---------------
import climeval

# Load datasets
model_ds = climeval.open_dataset_with_chunks("model.nc")
obs_ds = climeval.open_dataset_with_chunks("obs.nc")

# Compute metrics
metrics = climeval.compute_continuous_metrics(model_ds["variable"], obs_ds["variable"])
print(metrics)

# Visualize results
climeval.visualize_map(model_ds["variable"], obs_ds["variable"])

License: MIT
Author: Shiv Shankar Singh
Version: 0.1.1
"""

from .core import compute_continuous_metrics, compute_event_metrics, compute_gridwise_continuous
from .visualization import visualize_map, visualize_metric_map
from .utils import open_dataset_with_chunks, detect_or_fix_mismatch
from .dask_setup import init_dask_cluster

__version__ = "0.1.1"

# Define what is imported when using `from climeval import *`
__all__ = [
    "compute_continuous_metrics",
    "compute_event_metrics",
    "compute_gridwise_continuous",
    "visualize_map",
    "visualize_metric_map",
    "open_dataset_with_chunks",
    "detect_or_fix_mismatch",
    "init_dask_cluster"
]
