import os
import numpy as np
import xarray as xr
from typing import Dict, Tuple, Any, List

# Standard Oceanographic Parameters & Grid Specifications
LAT_MIN, LAT_MAX, LAT_STEP = 5.0, 30.0, 0.25
LON_MIN, LON_MAX, LON_STEP = 45.0, 105.0, 0.25

DEPTH_LEVELS = [0, 10, 20, 50, 75, 100, 150, 200, 300, 500, 750, 1000, 1250, 1500, 2000]

SURFACE_VARS = ["thetao", "zos", "so", "uo", "vo"]

# Standard Normalization Statistics (Mean, Std)
CHANNEL_STATS = {
    "thetao": {"mean": 28.2, "std": 1.8},     # SST in °C
    "zos": {"mean": 0.05, "std": 0.15},        # SSH in m
    "so": {"mean": 34.8, "std": 1.4},          # SSS in PSU
    "uo": {"mean": 0.12, "std": 0.45},         # Wind/Current U in m/s
    "vo": {"mean": 0.08, "std": 0.38},         # Wind/Current V in m/s
}

class OceanDataEngine:
    """
    Oceanographic Data Engine for North Indian Ocean Domain.
    Handles data ingestion, physical synthetic fallback generation,
    coordinate patch extraction for ML, and simulated ARGO sounding generation.
    """
    def __init__(self, data_dir: str = None):
        if data_dir is None:
            # Default to repo root /data/processed
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.data_dir = os.path.join(base_dir, "data", "processed")
        else:
            self.data_dir = data_dir

        os.makedirs(self.data_dir, exist_ok=True)
        self.surface_file = os.path.join(self.data_dir, "final_filtered_data.nc")
        self.target_file = os.path.join(self.data_dir, "final_filtered_target_data.nc")

        self.latitudes = np.arange(LAT_MIN, LAT_MAX + LAT_STEP / 2, LAT_STEP)
        self.longitudes = np.arange(LON_MIN, LON_MAX + LON_STEP / 2, LON_STEP)
        self.depths = np.array(DEPTH_LEVELS, dtype=np.float32)

        self.surface_ds = None
        self.target_ds = None
        
        self.load_or_generate_datasets()

    def generate_synthetic_datasets(self):
        """
        Generates calibrated, physical synthetic NetCDF datasets covering the
        North Indian Ocean (5°N–30°N, 45°E–105°E) with realistic ocean dynamics:
        - Arabian Sea high salinity (35.5 - 37 PSU) & summer upwelling cooling off Oman/Somalia.
        - Bay of Bengal freshwater pool (32 - 34 PSU) & warm cyclone heat pool (>29.5°C).
        - Mesoscale eddies (SSH anomalies +/-0.35m altering thermocline depth).
        - Sigmoidal thermocline profile across 15 standard depth layers.
        """
        if self.surface_ds is not None:
            self.surface_ds.close()
        if self.target_ds is not None:
            self.target_ds.close()

        print("[DataEngine] Generating physical synthetic NetCDF datasets for North Indian Ocean...")
        n_lat = len(self.latitudes)
        n_lon = len(self.longitudes)
        n_depth = len(self.depths)

        lon_grid, lat_grid = np.meshgrid(self.longitudes, self.latitudes)

        # 1. Sea Surface Temperature (thetao): 24°C - 31°C
        # Warmer in equatorial & eastern basin (Bay of Bengal ~29.5-31°C), cooler in NW Arabian Sea (~25-27°C)
        base_sst = 28.5 + 1.2 * np.sin(np.radians(lat_grid * 3.0)) + 0.8 * np.sin(np.radians((lon_grid - 60.0) * 2.5))
        # Add Bay of Bengal warm pool feature (Lon: 85-95, Lat: 10-18)
        bob_warm_pool = 1.6 * np.exp(-(((lat_grid - 14.5) / 5.0)**2 + ((lon_grid - 88.5) / 5.0)**2))
        # Add Arabian Sea cold eddy / upwelling off Oman/Somalia (Lon: 55-65, Lat: 15-22)
        arabian_upwelling = -1.8 * np.exp(-(((lat_grid - 18.0) / 4.0)**2 + ((lon_grid - 60.0) / 5.0)**2))
        # Add mesoscale turbulent wave pattern
        meso_waves = 0.4 * np.sin(lat_grid * 1.5) * np.cos(lon_grid * 1.5)
        sst = np.clip(base_sst + bob_warm_pool + arabian_upwelling + meso_waves, 24.0, 31.5)

        # 2. Sea Surface Height Anomaly (zos): -0.35m to +0.35m
        # Warm eddies have positive SSH (anticyclonic), cold upwelling eddies have negative SSH (cyclonic)
        ssh = 0.18 * (sst - np.mean(sst)) / np.std(sst) + 0.06 * np.sin(lat_grid * 2.0 + lon_grid * 1.2)
        ssh = np.clip(ssh, -0.35, 0.35)

        # 3. Sea Surface Salinity (so): 32 - 37 PSU
        # High salinity in Arabian Sea (Lon 50-75E: 35.5 - 36.8 PSU), low in Bay of Bengal (Lon 80-98E: 32.0 - 34.2 PSU)
        base_sss = 36.2 - 3.8 * (1.0 / (1.0 + np.exp(-(lon_grid - 78.0) / 3.0)))
        sss_noise = 0.25 * np.sin(lat_grid * 2.2) * np.cos(lon_grid * 1.8)
        sss = np.clip(base_sss + sss_noise, 32.0, 37.0)

        # 4. Wind/Current vectors: uo (eastward), vo (northward) in m/s
        uo = 0.35 * np.cos(np.radians(lat_grid * 4.0)) + 0.2 * np.sin(np.radians(lon_grid * 3.0))
        vo = 0.30 * np.sin(np.radians(lat_grid * 3.5)) - 0.15 * np.cos(np.radians(lon_grid * 2.5))

        # Build 2D Surface Dataset
        surface_ds = xr.Dataset(
            data_vars={
                "thetao": (["latitude", "longitude"], sst.astype(np.float32), {"units": "degrees_C", "long_name": "Sea Surface Temperature"}),
                "zos": (["latitude", "longitude"], ssh.astype(np.float32), {"units": "m", "long_name": "Sea Surface Height Anomaly"}),
                "so": (["latitude", "longitude"], sss.astype(np.float32), {"units": "PSU", "long_name": "Sea Surface Salinity"}),
                "uo": (["latitude", "longitude"], uo.astype(np.float32), {"units": "m/s", "long_name": "Eastward Surface Velocity/Wind"}),
                "vo": (["latitude", "longitude"], vo.astype(np.float32), {"units": "m/s", "long_name": "Northward Surface Velocity/Wind"}),
            },
            coords={
                "latitude": self.latitudes,
                "longitude": self.longitudes,
            },
            attrs={"description": "OceanEmbed Synthetic Surface Physics Reanalysis - North Indian Ocean"}
        )

        if os.path.exists(self.surface_file):
            try:
                os.remove(self.surface_file)
            except Exception:
                pass
        surface_ds.to_netcdf(self.surface_file)
        print(f"[DataEngine] Saved calibrated surface dataset -> {self.surface_file}")

        # 5. Build 3D Subsurface Temperature Target Dataset (15 depth layers)
        t_3d = np.zeros((n_depth, n_lat, n_lon), dtype=np.float32)
        t_deep = 2.8 # Abyssal ocean temperature (°C) at 2000m

        z_center = 120.0 + 60.0 * (ssh / 0.35) - 15.0 * np.cos(np.radians(lat_grid * 2.5))
        z_center = np.clip(z_center, 60.0, 180.0)

        scale = 55.0 + 10.0 * (sst - 28.0) / 3.0
        scale = np.clip(scale, 40.0, 75.0)

        f_0 = 1.0 / (1.0 + np.exp(-z_center / scale))

        for k, z in enumerate(self.depths):
            if z == 0:
                t_3d[k, :, :] = sst
            else:
                arg = np.clip((z - z_center) / scale, -10.0, 15.0)
                f_z = 1.0 / (1.0 + np.exp(arg))
                decay_fraction = f_z / f_0
                layer_temp = t_deep + (sst - t_deep) * decay_fraction
                strat_noise = 0.03 * np.sin(z / 50.0 + lat_grid * 0.4)
                t_3d[k, :, :] = np.clip(layer_temp + strat_noise, t_deep, sst)

        target_ds = xr.Dataset(
            data_vars={
                "thetao": (["depth", "latitude", "longitude"], t_3d, {"units": "degrees_C", "long_name": "3D Ocean Potential Temperature"})
            },
            coords={
                "depth": self.depths,
                "latitude": self.latitudes,
                "longitude": self.longitudes,
            },
            attrs={"description": "OceanEmbed 3D Subsurface Temperature Target Dataset (0-2000m)"}
        )

        if os.path.exists(self.target_file):
            try:
                os.remove(self.target_file)
            except Exception:
                pass
        target_ds.to_netcdf(self.target_file)
        print(f"[DataEngine] Saved calibrated 3D target dataset -> {self.target_file}")

        self.surface_ds = surface_ds.load()
        self.target_ds = target_ds.load()

    def load_or_generate_datasets(self):
        """Loads processed NetCDF files or triggers fallback generation."""
        if not os.path.exists(self.surface_file) or not os.path.exists(self.target_file):
            self.generate_synthetic_datasets()
        else:
            try:
                print(f"[DataEngine] Loading existing NetCDF datasets from {self.data_dir}...")
                with xr.open_dataset(self.surface_file) as ds_s:
                    self.surface_ds = ds_s.load()
                with xr.open_dataset(self.target_file) as ds_t:
                    self.target_ds = ds_t.load()
                # Verify coordinates
                self.latitudes = self.surface_ds.latitude.values
                self.longitudes = self.surface_ds.longitude.values
                # Do not overwrite self.depths from file so it interpolates to 15 standard levels
                print("[DataEngine] Successfully loaded active datasets into memory.")
            except Exception as e:
                print(f"[DataEngine] Failed to load existing files ({e}). Regenerating...")
                self.generate_synthetic_datasets()

    def get_nearest_indices(self, lat: float, lon: float) -> Tuple[int, int]:
        """Returns the nearest grid indices (lat_idx, lon_idx) for given coordinates."""
        lat_clamped = np.clip(lat, LAT_MIN, LAT_MAX)
        lon_clamped = np.clip(lon, LON_MIN, LON_MAX)

        lat_idx = int(np.abs(self.latitudes - lat_clamped).argmin())
        lon_idx = int(np.abs(self.longitudes - lon_clamped).argmin())
        return lat_idx, lon_idx

    def extract_patch(self, lat: float, lon: float, patch_size: int = 16) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Extracts a (C, H, W) normalized tensor patch centered at (lat, lon) for ML inference.
        C = 5 (thetao, zos, so, uo, vo).
        Also returns raw surface variables at query coordinate.
        """
        lat_idx, lon_idx = self.get_nearest_indices(lat, lon)
        half = patch_size // 2

        n_lat = len(self.latitudes)
        n_lon = len(self.longitudes)

        # Slice bounds with padding handling
        lat_start = max(0, lat_idx - half)
        lat_end = min(n_lat, lat_idx + half)
        lon_start = max(0, lon_idx - half)
        lon_end = min(n_lon, lon_idx + half)

        patch = np.zeros((len(SURFACE_VARS), patch_size, patch_size), dtype=np.float32)
        raw_surface_vals = {}

        for c_idx, var_name in enumerate(SURFACE_VARS):
            if var_name in self.surface_ds:
                var_data = self.surface_ds[var_name].values
                if var_data.ndim == 3:
                    var_data = var_data[0] # Take first time step
            else:
                var_data = np.zeros((n_lat, n_lon), dtype=np.float32)

            sub_grid = var_data[lat_start:lat_end, lon_start:lon_end]
            
            # Pad to (patch_size, patch_size) if near grid boundaries
            pad_h_before = (lat_idx - half) * -1 if (lat_idx - half) < 0 else 0
            pad_h_after = (lat_idx + half) - n_lat if (lat_idx + half) > n_lat else 0
            pad_w_before = (lon_idx - half) * -1 if (lon_idx - half) < 0 else 0
            pad_w_after = (lon_idx + half) - n_lon if (lon_idx + half) > n_lon else 0

            padded = np.pad(
                sub_grid,
                ((pad_h_before, pad_h_after), (pad_w_before, pad_w_after)),
                mode="edge"
            )

            # Ensure exact patch dimensions
            padded = padded[:patch_size, :patch_size]

            # Normalize using standard statistics
            mean_val = CHANNEL_STATS[var_name]["mean"]
            std_val = CHANNEL_STATS[var_name]["std"]
            norm_patch = (padded - mean_val) / std_val
            
            # Handle land masks (NaNs)
            norm_patch = np.nan_to_num(norm_patch, nan=0.0)

            patch[c_idx] = norm_patch
            raw_val = float(var_data[lat_idx, lon_idx])
            raw_surface_vals[var_name] = 0.0 if np.isnan(raw_val) else raw_val

        return patch, raw_surface_vals

    def get_ground_truth_profile(self, lat: float, lon: float) -> np.ndarray:
        """Returns the actual 15-point ground truth temperature profile from dataset."""
        lat_idx, lon_idx = self.get_nearest_indices(lat, lon)
        if "thetao" in self.target_ds:
            vals = self.target_ds["thetao"].values
            if vals.ndim == 4:
                gt_profile = vals[0, :, lat_idx, lon_idx]
            else:
                gt_profile = vals[:, lat_idx, lon_idx]
                
            if "depth" in self.target_ds:
                file_depths = self.target_ds.depth.values
                if len(file_depths) != len(self.depths):
                    gt_profile = np.interp(self.depths, file_depths, gt_profile)
                    
            # Handle land masks (NaNs)
            gt_profile = np.nan_to_num(gt_profile, nan=2.8)
        else:
            sst = float(self.surface_ds["thetao"].values[lat_idx, lon_idx])
            z_center = 120.0
            scale = 55.0
            f_0 = 1.0 / (1.0 + np.exp(-z_center / scale))
            decay = (1.0 / (1.0 + np.exp((self.depths - z_center) / scale))) / f_0
            gt_profile = 2.8 + (sst - 2.8) * decay
        return np.array(gt_profile, dtype=np.float32)

    def generate_simulated_argo_sounding(self, lat: float, lon: float, gt_profile: np.ndarray) -> np.ndarray:
        """
        Simulates an in-situ ARGO profiling float sounding matching the location,
        incorporating realistic Seabird CTD sensor calibration noise (~0.05°C - 0.15°C).
        """
        np.random.seed(int((lat * 100 + lon * 10) % 100000))
        noise = np.random.normal(0.0, 0.08, size=gt_profile.shape)
        # Surface noise is minimal
        noise[0] = np.random.normal(0.0, 0.03)
        argo_profile = gt_profile + noise
        
        # Enforce physically valid ocean temperature decay
        for i in range(1, len(argo_profile)):
            if argo_profile[i] > argo_profile[i - 1]:
                argo_profile[i] = argo_profile[i - 1] - 0.01
        return np.clip(argo_profile, 2.0, 32.0)

    def get_surface_grid_summary(self, step: int = 2) -> Dict[str, Any]:
        """
        Returns downsampled 2D surface fields for fast map visualization in frontend.
        """
        sub_lats = self.latitudes[::step].tolist()
        sub_lons = self.longitudes[::step].tolist()

        sst_grid = self.surface_ds["thetao"].values[::step, ::step].tolist()
        ssh_grid = self.surface_ds["zos"].values[::step, ::step].tolist()
        sss_grid = self.surface_ds["so"].values[::step, ::step].tolist()

        return {
            "latitudes": sub_lats,
            "longitudes": sub_lons,
            "sst": sst_grid,
            "ssh": ssh_grid,
            "sss": sss_grid,
            "bounds": {
                "minLat": LAT_MIN,
                "maxLat": LAT_MAX,
                "minLon": LON_MIN,
                "maxLon": LON_MAX,
            }
        }

# Global Data Engine Instance
data_engine = OceanDataEngine()

