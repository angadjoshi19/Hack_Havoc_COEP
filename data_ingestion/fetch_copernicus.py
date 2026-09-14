import os
import copernicusmarine
from dotenv import load_dotenv

def fetch_data(output_dir="../data/raw"):
    """
    Fetches required parameters from Copernicus Marine Service:
    - Sea Surface Temperature (SST)
    - Sea Surface Salinity (SSS)
    - Sea Surface Height/Anomaly (SSH/SLA)
    - Surface Currents (U and V vectors)
    - Surface Winds (U and V vectors)
    """
    # Load credentials
    load_dotenv()
    username = os.getenv("COPERNICUS_USERNAME")
    password = os.getenv("COPERNICUS_PASSWORD")
    
    if not username or not password or username == "your_username_here":
        raise ValueError("Please update COPERNICUS_USERNAME and COPERNICUS_PASSWORD in the .env file")
    
    # Ensure output directory exists relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, output_dir)
    os.makedirs(output_path, exist_ok=True)
    
    # Standard dataset IDs (Analysis and Forecast)
    # Note: These are standard globally used IDs. 
    # physics: GLOBAL_ANALYSISFORECAST_PHY_001_024
    # winds: WIND_GLO_WIND_L4_NRT_OBSERVATIONS_012_004
    
    # We will use the GLORYS12V1 Reanalysis Product for all Physics data
    # Product ID: GLOBAL_REANALYSIS_PHY_001_030
    dataset_ids = {
        "physics": "cmems_mod_glo_phy_my_0.083deg_P1D-m", # Daily mean unified physics dataset
        "winds": "cmems_obs-wind_glo_phy_my_l4_0.125deg_P1D" # Reanalysis wind
    }
    
    print("Fetching Physics Data (SST, SSS, SSH, Currents)...")
    try:
        copernicusmarine.subset(
            dataset_id=dataset_ids["physics"],
            variables=["thetao", "so", "zos", "uo", "vo"], # thetao: SST, so: SSS, zos: SSH, uo/vo: currents
            start_datetime="2023-01-01T00:00:00",
            end_datetime="2023-01-02T00:00:00",
            minimum_longitude=45.0,
            maximum_longitude=105.0,
            minimum_latitude=5.0,
            maximum_latitude=30.0,
            minimum_depth=0.0,
            maximum_depth=0.5, # Surface layer only
            output_directory=output_path,
            output_filename="physics_data.nc",
            username=username,
            password=password,
            force_download=True
        )
        print("    [SUCCESS] Physics surface data fetched.")
    except Exception as e:
        print(f"    [FAILED] Error fetching physics data: {e}")

    print("\nFetching Physics Target Data (3D Subsurface Temperature to 4000m)...")
    try:
        copernicusmarine.subset(
            dataset_id=dataset_ids["physics"],
            variables=["thetao"], 
            start_datetime="2023-01-01T00:00:00",
            end_datetime="2023-01-02T00:00:00",
            minimum_longitude=45.0,
            maximum_longitude=105.0,
            minimum_latitude=5.0,
            maximum_latitude=30.0,
            minimum_depth=0.0,
            maximum_depth=4000.0, # Full depth profile
            output_directory=output_path,
            output_filename="physics_target_data.nc",
            username=username,
            password=password,
            force_download=True
        )
        print("    [SUCCESS] 3D Target data fetched.")
    except Exception as e:
        print(f"    [FAILED] Error fetching 3D target data: {e}")

    print("\nFetching Wind Data (U and V vectors)...")
    try:
        copernicusmarine.subset(
            dataset_id=dataset_ids["winds"],
            variables=["eastward_wind", "northward_wind"], 
            start_datetime="2023-01-01T00:00:00",
            end_datetime="2023-01-02T00:00:00",
            minimum_longitude=45.0,
            maximum_longitude=105.0,
            minimum_latitude=5.0,
            maximum_latitude=30.0,
            output_directory=output_path,
            output_filename="wind_data.nc",
            username=username,
            password=password,
            force_download=True
        )
        print("    [SUCCESS] Wind data fetched.")
    except Exception as e:
        print(f"    [FAILED] Error fetching wind data: {e}")

if __name__ == "__main__":
    print("Starting Copernicus Marine Data Fetch...")
    fetch_data()
