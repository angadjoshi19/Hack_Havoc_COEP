# OceanEmbed: Data Ingestion Phase

This document explains the technical architecture and pipeline for the data ingestion phase of the OceanEmbed project. The goal of this phase is to fetch, harmonize, and standardize satellite observations and 3D subsurface data to be used as tensors for the PyTorch deep learning models.

## Pipeline Overview

The data ingestion pipeline consists of four main Python scripts located in the `data_ingestion/` module. The pipeline is designed to be run sequentially.

### 1. `pipeline.py` (The Orchestrator)
This is the main entry point for the data ingestion phase. It uses the `subprocess` module to sequentially execute the fetch, preprocess, and filtering scripts. If any phase fails, the orchestrator halts execution to prevent corrupted data from propagating.

### 2. `fetch_copernicus.py` (Data Fetching)
This script utilizes the `copernicusmarine` API client to securely fetch multi-dimensional NetCDF datasets.
- **Surface Predictors**: It extracts Sea Surface Temperature (SST), Sea Surface Salinity (SSS), Sea Surface Height (SSH), and Surface Currents (U and V) from the physics dataset, and Surface Winds (U and V) from the wind dataset. These are bounded strictly to the surface layer (`depth 0.0 - 0.5m`).
- **3D Ground Truth**: It makes a separate API call to extract the 3D subsurface temperature (`thetao`) down to a depth of `4000m` to serve as the ground truth training target for the AI engine.
- **Geographic Bounding**: All downloaded data is heavily constrained to the **North Indian Ocean (5°N to 30°N and 45°E to 105°E)** to focus training and minimize raw payload sizes.

### 3. `preprocess.py` (Harmonization)
Due to the different sources of satellite data (e.g., wind sensors vs. thermal sensors), raw datasets often have misaligned spatial grids.
This script uses `xarray` to:
- Load the raw `.nc` files.
- Establish a uniform, standardized **0.25° × 0.25°** spatial grid covering the North Indian Ocean.
- Interpolate all disparate datasets (surface parameters and 3D targets) onto this strict common coordinate system.
- Merge the 2D surface predictors into a single cohesive dataset.

### 4. `statistical_filtering.py` (Filtering & Standardization)
Machine learning models are highly sensitive to NaN values and wild outliers. This script utilizes `scipy` and `numpy` to clean the unified dataset:
- **Probabilistic Gap Filling**: It interpolates over very small spatial gaps (limit of 2 pixels) using linear interpolation to fill small sensor occlusions over the ocean without artificially interpolating over landmasses.
- **Anomaly Capping**: It computes robust statistics and caps extreme outliers (e.g., beyond 4 standard deviations) that usually represent sensor glitches or retrieval errors.
- **Standardization**: It performs a Z-score normalization, forcing every variable into a standard normal distribution (`mean = 0, variance = 1`). This is a mathematically crucial step to ensure the CNN+ViT pipeline converges smoothly during training.

## How to Run

1.  Ensure you have created a `.env` file in the root directory containing your `COPERNICUS_USERNAME` and `COPERNICUS_PASSWORD`.
2.  Install dependencies: `pip install -r data_ingestion/requirements.txt`
3.  Execute the pipeline: `python data_ingestion/pipeline.py`

Raw files are temporarily stored in `data/raw/` while the final tensor-ready training files are generated in `data/processed/`.
