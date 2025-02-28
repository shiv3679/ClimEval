# app.py

import streamlit as st
import xarray as xr
import io
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Import your backend functions from 'backend.py' (adjust the file name if needed).
from backend import (
    init_dask_cluster,
    open_dataset_with_chunks,
    detect_or_fix_mismatch,
    compute_continuous_metrics,
    compute_event_metrics,
    compute_probabilistic_metrics,
    compute_distribution_metrics,
    compute_gridwise_continuous,
    compute_gridwise_event_metrics
)

def main():
    st.title("EvalMetrics: Precision in Prediction")
    st.write("Use this interface to load your NetCDF files, choose metrics, and visualize results.")

    # 1. GPU usage
    use_gpu = st.checkbox("Use GPU acceleration? (Requires CuPy)", value=False)
    # 2. Dask workers
    n_workers = st.number_input("Number of Dask Workers", min_value=1, max_value=64, value=1)

    # 3. File uploads
    st.subheader("Upload Your NetCDF Datasets")
    model_file = st.file_uploader("Model NetCDF", type=["nc"])
    obs_file   = st.file_uploader("Observation NetCDF", type=["nc"])
    
    # 4. Chunk sizes
    st.subheader("Chunk Sizes")
    chunk_time = st.number_input("Time chunk size", value=10)
    chunk_lat  = st.number_input("Lat chunk size", value=200)
    chunk_lon  = st.number_input("Lon chunk size", value=200)
    chunks = {"time": chunk_time, "lat": chunk_lat, "lon": chunk_lon}

    # We'll store objects in Streamlit session_state to avoid re-loading repeatedly
    if "model_ds" not in st.session_state:
        st.session_state["model_ds"] = None
    if "obs_ds" not in st.session_state:
        st.session_state["obs_ds"] = None
    if "client" not in st.session_state:
        st.session_state["client"] = None

    # Button to load datasets
    if st.button("Load & (Optionally) Regrid"):
        if model_file is not None and obs_file is not None:
            # 1) Initialize Dask cluster
            st.session_state["client"] = init_dask_cluster(use_gpu=use_gpu, n_workers=n_workers)
            
            # 2) Convert the uploaded files (BytesIO) for xarray
            model_bytes = model_file.read()
            obs_bytes   = obs_file.read()
            model_ds = open_dataset_with_chunks(io.BytesIO(model_bytes), chunks=chunks, use_gpu=use_gpu)
            obs_ds   = open_dataset_with_chunks(io.BytesIO(obs_bytes),   chunks=chunks, use_gpu=use_gpu)

            # 3) Check dimension mismatch, possibly regrid
            model_ds, obs_ds = detect_or_fix_mismatch(model_ds, obs_ds)

            # Store in session
            st.session_state["model_ds"] = model_ds
            st.session_state["obs_ds"]   = obs_ds
            st.success("Datasets loaded, regridding done if needed!")
        else:
            st.warning("Please upload both Model and Observation NetCDF files first.")

    # If we have loaded DS, we proceed
    if st.session_state["model_ds"] is not None and st.session_state["obs_ds"] is not None:
        model_ds = st.session_state["model_ds"]
        obs_ds   = st.session_state["obs_ds"]

        st.subheader("Variable Selection & Domain-Wide Metrics")

        # 1) Let user pick a variable
        model_vars = list(model_ds.data_vars.keys())
        var_choice = st.selectbox("Select variable from Model DS", model_vars)
        # Ensure the same var exists in obs
        if var_choice not in obs_ds.data_vars:
            st.error(f"Variable '{var_choice}' not found in Observation DS.")
            st.stop()

        model_da = model_ds[var_choice]
        obs_da   = obs_ds[var_choice]

        # 2) Choose domain-wide metrics
        st.write("Check the metrics you want to compute:")
        cont_check = st.checkbox("Continuous (MSE, MAE, Bias, Corr)")
        event_check= st.checkbox("Event-based (POD, FAR, CSI, ETS, FSS)")
        prob_check = st.checkbox("Probabilistic (Brier Score, etc.)")
        dist_check = st.checkbox("Distribution-level (Wasserstein, Jensen-Shannon)")

        if st.button("Compute Domain-Wide Metrics"):
            if cont_check:
                cont_res = compute_continuous_metrics(model_da, obs_da)
                st.write("**Continuous Metrics**:", cont_res)
            if event_check:
                threshold_val = st.number_input("Event Threshold", value=1.0)
                evt_res = compute_event_metrics(model_da, obs_da, threshold=threshold_val)
                st.write("**Event-Based Metrics**:", evt_res)
            if prob_check:
                st.write("Assuming model variable is probability (0..1). If it's not, results may be meaningless.")
                obs_thr = st.number_input("Threshold to convert Obs to 0/1", value=0.5)
                obs_event = (obs_da >= obs_thr).astype(int)
                prob_res = compute_probabilistic_metrics(model_da, obs_event)
                st.write("**Probabilistic Metrics**:", prob_res)
            if dist_check:
                dist_res = compute_distribution_metrics(model_da, obs_da)
                st.write("**Distribution-Level Metrics**:", dist_res)

        st.subheader("Optional: Per-Pixel (Gridwise) Metrics")
        do_gridwise = st.checkbox("Compute & visualize per-pixel (gridwise) metrics?", value=False)

        if do_gridwise:
            if st.button("Compute & Visualize Gridwise Metrics"):
                # A) Continuous
                st.write("**Computing Continuous Gridwise Metrics** (Bias, MAE, MSE, Corr) ...")
                grid_cont = compute_gridwise_continuous(model_da, obs_da).compute()
                st.write("Done computing gridwise continuous metrics. Displaying maps below...")

                # 1) Bias Map
                st.write("**Bias Map**")
                fig_b, ax_b = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()}, figsize=(5,4))
                ax_b.set_title("Bias Map")
                ax_b.coastlines()
                ax_b.add_feature(cfeature.BORDERS, linestyle=':')
                bias_data = grid_cont["bias_map"]
                bias_data.plot(ax=ax_b, x="lon", y="lat", transform=ccrs.PlateCarree(), cmap="RdBu", center=0)
                st.pyplot(fig_b)

                # 2) MAE Map
                st.write("**MAE Map**")
                fig_mae, ax_mae = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()}, figsize=(5,4))
                ax_mae.set_title("MAE Map")
                ax_mae.coastlines()
                ax_mae.add_feature(cfeature.BORDERS, linestyle=':')
                mae_data = grid_cont["mae_map"]
                mae_data.plot(ax=ax_mae, x="lon", y="lat", transform=ccrs.PlateCarree(), cmap="Reds")
                st.pyplot(fig_mae)

                # 3) MSE Map
                st.write("**MSE Map**")
                fig_mse, ax_mse = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()}, figsize=(5,4))
                ax_mse.set_title("MSE Map")
                ax_mse.coastlines()
                ax_mse.add_feature(cfeature.BORDERS, linestyle=':')
                mse_data = grid_cont["mse_map"]
                mse_data.plot(ax=ax_mse, x="lon", y="lat", transform=ccrs.PlateCarree(), cmap="Reds")
                st.pyplot(fig_mse)

                # 4) Correlation Map
                st.write("**Correlation Map**")
                fig_c, ax_c = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()}, figsize=(5,4))
                ax_c.set_title("Correlation Map")
                ax_c.coastlines()
                ax_c.add_feature(cfeature.BORDERS, linestyle=':')
                corr_data = grid_cont["corr_map"]
                corr_data.plot(ax=ax_c, x="lon", y="lat", transform=ccrs.PlateCarree(), cmap="RdBu", center=0)
                st.pyplot(fig_c)

                st.write("**Continuous metric maps displayed!**")

                # B) Event-based
                threshold_val_2 = st.number_input("Threshold for Gridwise Event-based metrics?", value=1.0)
                st.write("**Computing Gridwise Event-based Metrics** (POD, FAR, CSI, ETS) ...")
                grid_evt = compute_gridwise_event_metrics(model_da, obs_da, threshold=threshold_val_2).compute()
                st.write("Done computing event-based gridwise. Displaying selected map(s) below...")

                # We show POD map as an example. You can replicate for FAR_map, CSI_map, ETS_map as well.
                st.write("**POD Map**")
                fig_pod, ax_pod = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()}, figsize=(5,4))
                ax_pod.set_title("POD Map")
                ax_pod.coastlines()
                ax_pod.add_feature(cfeature.BORDERS, linestyle=':')
                grid_evt["POD_map"].plot(ax=ax_pod, x="lon", y="lat", transform=ccrs.PlateCarree(), cmap="Blues")
                st.pyplot(fig_pod)

                # If you also want to show FAR, CSI, ETS:
                st.write("**FAR Map**")
                fig_far, ax_far = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()}, figsize=(5,4))
                ax_far.set_title("FAR Map")
                ax_far.coastlines()
                ax_far.add_feature(cfeature.BORDERS, linestyle=':')
                grid_evt["FAR_map"].plot(ax=ax_far, x="lon", y="lat", transform=ccrs.PlateCarree(), cmap="Reds")
                st.pyplot(fig_far)

                st.write("**CSI Map**")
                fig_csi, ax_csi = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()}, figsize=(5,4))
                ax_csi.set_title("CSI Map")
                ax_csi.coastlines()
                ax_csi.add_feature(cfeature.BORDERS, linestyle=':')
                grid_evt["CSI_map"].plot(ax=ax_csi, x="lon", y="lat", transform=ccrs.PlateCarree(), cmap="Greens")
                st.pyplot(fig_csi)

                st.write("**ETS Map**")
                fig_ets, ax_ets = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()}, figsize=(5,4))
                ax_ets.set_title("ETS Map")
                ax_ets.coastlines()
                ax_ets.add_feature(cfeature.BORDERS, linestyle=':')
                grid_evt["ETS_map"].plot(ax=ax_ets, x="lon", y="lat", transform=ccrs.PlateCarree(), cmap="RdBu", center=0)
                st.pyplot(fig_ets)

                st.success("All per-pixel maps displayed. Feel free to scroll and explore each one!")


def run_app():
    main()

if __name__ == "__main__":
    run_app()
