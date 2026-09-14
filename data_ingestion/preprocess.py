import os
import xarray as xr
import numpy as np

def preprocess_data(raw_dir="../data/raw", processed_dir="../data/processed"):
    """
    Loads raw NetCDF datasets, handles dimensional differences, 
    interpolates to a common grid, and merges them.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    raw_path = os.path.join(script_dir, raw_dir)
    processed_path = os.path.join(script_dir, processed_dir)
    os.makedirs(processed_path, exist_ok=True)
    
    physics_file = os.path.join(raw_path, "physics_data.nc")
    wind_file = os.path.join(raw_path, "wind_data.nc")
    target_file = os.path.join(raw_path, "physics_target_data.nc")
    
    datasets_to_merge = []
    
    # Define target 0.25 x 0.25 degree grid for the North Indian Ocean
    target_lon = np.arange(45.0, 105.0 + 0.25, 0.25)
    target_lat = np.arange(5.0, 30.0 + 0.25, 0.25)
    target_ds = xr.Dataset(
        {
            "latitude": (["latitude"], target_lat),
            "longitude": (["longitude"], target_lon),
        }
    )
    
    print("Loading raw NetCDF datasets...")
    if os.path.exists(physics_file):
        ds_phys = xr.open_dataset(physics_file)
        # Drop depth dimension if it exists and is size 1 (surface layer only)
        if 'depth' in ds_phys.dims:
            ds_phys = ds_phys.squeeze('depth', drop=True)
        print(f"Loaded physics data: {list(ds_phys.data_vars)}")
        
        print("Interpolating physics data to 0.25° grid...")
        ds_phys_interp = ds_phys.interp(
            latitude=target_ds.latitude,
            longitude=target_ds.longitude,
            method="nearest",
            kwargs={"fill_value": "extrapolate"}
        )
        datasets_to_merge.append(ds_phys_interp)
    else:
        print(f"Warning: {physics_file} not found. Ensure fetch_copernicus.py ran successfully.")
        
    if os.path.exists(wind_file):
        ds_wind = xr.open_dataset(wind_file)
        print(f"Loaded wind data: {list(ds_wind.data_vars)}")
        
        print("Interpolating wind data to 0.25° grid...")
        ds_wind_interp = ds_wind.interp(
            latitude=target_ds.latitude,
            longitude=target_ds.longitude,
            method="nearest",
            kwargs={"fill_value": "extrapolate"}
        )
        datasets_to_merge.append(ds_wind_interp)
    else:
        print(f"Warning: {wind_file} not found. Ensure fetch_copernicus.py ran successfully.")
        
    if not datasets_to_merge:
        print("No datasets to process.")
        return
        
    print("Merging datasets...")
    try:
        # Merge datasets using xarray
        ds_merged = xr.merge(datasets_to_merge)
        
        # Save preprocessed output
        output_file = os.path.join(processed_path, "merged_surface_data.nc")
        ds_merged.to_netcdf(output_file)
        print(f"Preprocessing complete. Merged surface data saved to {output_file}")
    except Exception as e:
        print(f"Error during merging/preprocessing: {e}")
        
    print("Processing 3D target data...")
    if os.path.exists(target_file):
        ds_target = xr.open_dataset(target_file)
        print(f"Loaded target data: {list(ds_target.data_vars)}")
        
        print("Interpolating target data to 0.25° grid...")
        ds_target_interp = ds_target.interp(
            latitude=target_ds.latitude,
            longitude=target_ds.longitude,
            method="nearest",
            kwargs={"fill_value": "extrapolate"}
        )
        
        output_target_file = os.path.join(processed_path, "processed_target_data.nc")
        ds_target_interp.to_netcdf(output_target_file)
        print(f"Target data processed and saved to {output_target_file}")
    else:
        print(f"Warning: {target_file} not found. Ensure fetch_copernicus.py ran successfully.")

if __name__ == "__main__":
    preprocess_data()
