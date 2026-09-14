import os
import xarray as xr
import numpy as np
from scipy import stats

def filter_and_clean(processed_dir="../data/processed", output_dir="../data/processed"):
    """
    Performs statistical filtering, missing data interpolation, and outlier capping.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    processed_path = os.path.join(script_dir, processed_dir)
    output_path = os.path.join(script_dir, output_dir)
    
    input_file = os.path.join(processed_path, "merged_surface_data.nc")
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found. Run preprocess.py first.")
        return
        
    print(f"Loading merged dataset from {input_file}...")
    ds = xr.open_dataset(input_file)
    
    print("Performing probabilistic gap filling (interpolation)...")
    # Interpolate small gaps over spatial dimensions (limit to 2 pixels to avoid interpolating over land masses)
    ds_filled = ds.interpolate_na(dim='longitude', method='linear', limit=2)
    ds_filled = ds_filled.interpolate_na(dim='latitude', method='linear', limit=2)
    
    print("Performing anomaly detection and capping outliers (Z-score)...")
    for var in ds_filled.data_vars:
        if np.issubdtype(ds_filled[var].dtype, np.number):
            # Calculate robust statistics
            mean_val = ds_filled[var].mean().item()
            std_val = ds_filled[var].std().item()
            
            # Define limits (e.g. 4 std deviations)
            threshold = 4
            lower_bound = mean_val - (threshold * std_val)
            upper_bound = mean_val + (threshold * std_val)
            
            # Cap the extreme outliers
            clipped = ds_filled[var].clip(min=lower_bound, max=upper_bound)
            
            # Standardize the data (mean=0, variance=1)
            # Recompute mean and std after clipping for accurate standardization
            new_mean = clipped.mean().item()
            new_std = clipped.std().item()
            ds_filled[var] = (clipped - new_mean) / (new_std if new_std != 0 else 1.0)
            
    output_file = os.path.join(output_path, "final_filtered_data.nc")
    ds_filled.to_netcdf(output_file)
    print(f"Surface data filtering complete. Final tensor-ready dataset saved to {output_file}")
    
    target_file = os.path.join(processed_path, "processed_target_data.nc")
    if os.path.exists(target_file):
        print(f"Loading target dataset from {target_file}...")
        ds_target = xr.open_dataset(target_file)
        
        print("Performing gap filling on target data...")
        ds_target_filled = ds_target.interpolate_na(dim='longitude', method='linear', limit=2)
        ds_target_filled = ds_target_filled.interpolate_na(dim='latitude', method='linear', limit=2)
        
        print("Standardizing target data (Z-score)...")
        for var in ds_target_filled.data_vars:
            if np.issubdtype(ds_target_filled[var].dtype, np.number):
                mean_val = ds_target_filled[var].mean().item()
                std_val = ds_target_filled[var].std().item()
                
                threshold = 4
                lower_bound = mean_val - (threshold * std_val)
                upper_bound = mean_val + (threshold * std_val)
                
                clipped = ds_target_filled[var].clip(min=lower_bound, max=upper_bound)
                new_mean = clipped.mean().item()
                new_std = clipped.std().item()
                ds_target_filled[var] = (clipped - new_mean) / (new_std if new_std != 0 else 1.0)
                
        output_target = os.path.join(output_path, "final_filtered_target_data.nc")
        ds_target_filled.to_netcdf(output_target)
        print(f"Target data filtering complete. Final tensor-ready target dataset saved to {output_target}")
    else:
        print(f"Warning: {target_file} not found.")

if __name__ == "__main__":
    filter_and_clean()
