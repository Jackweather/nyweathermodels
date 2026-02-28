import os
import requests
from datetime import datetime, timedelta
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patheffects as path_effects
import time
import gc
from filelock import FileLock
import cartopy
import importlib
import shutil
import json
from pyproj import Transformer
from scipy.ndimage import gaussian_filter

# Set base directory for HRRR output
BASE_DIR = '/var/data'

grib_dir = os.path.join(BASE_DIR, "GFS_frzr_surface", "grib")
png_dir = os.path.join(BASE_DIR, "GFS_frzr_surface", "png")
os.makedirs(grib_dir, exist_ok=True)
os.makedirs(png_dir, exist_ok=True)

forecast_steps = list(range(6, 385, 6))

# Define levels and colormap for freezing rain rate
frzr_levels = np.arange(0, 1.1, 0.1)  # Freezing rain rate levels in mm/hr
custom_cmap = LinearSegmentedColormap.from_list("frzr_cmap", [
    '#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#084594'
])
frzr_norm = BoundaryNorm(frzr_levels, custom_cmap.N)

# Define levels and colormap for probability of freezing precipitation
cpofp_levels = np.arange(0, 1.1, 0.1)  # Probability levels
cpofp_cmap = LinearSegmentedColormap.from_list("cpofp_cmap", [
    '#fff5eb', '#fee6ce', '#fdd0a2', '#fdae6b', '#fd8d3c', '#f16913', '#d94801', '#a63603', '#7f2704'
])
cpofp_norm = BoundaryNorm(cpofp_levels, cpofp_cmap.N)

# Define levels and colormap for temperature
tmp_levels = np.arange(-30, 41, 5)  # Temperature levels from -30°C to 40°C
tmp_cmap = LinearSegmentedColormap.from_list("tmp_cmap", [
    '#8e0152', '#c51b7d', '#de77ae', '#f1b6da', '#fde0ef', '#e6f5d0', '#a1d76a', '#4d9221', '#276419'
])

# Clear all files in the GRIB and PNG directories before starting
for folder in [grib_dir, png_dir]:
    for f in os.listdir(folder):
        file_path = os.path.join(folder, f)
        if os.path.isfile(file_path):
            os.remove(file_path)
print("Cleared GRIB and PNG directories before starting.")

def download_grib(url, file_path):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        print(f"Downloaded {os.path.basename(file_path)}")
        return file_path
    else:
        print(f"Failed to download {os.path.basename(file_path)} (Status Code: {response.status_code})")
        return None

def get_frzr_surface_grib(step):
    file_name = f"gfs.t{hour_str}z.pgrb2.0p25.f{step:03d}"
    file_path = os.path.join(grib_dir, f"frzr_surface_{file_name}")
    url = (
        f"https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?file={file_name}"
        f"&lev_surface=on&var_CFRZR=on"
        f"&dir=%2Fgfs.{date_str}%2F{hour_str}%2Fatmos"
    )
    return download_grib(url, file_path)

def get_cpofp_surface_grib(step):
    file_name = f"gfs.t{hour_str}z.pgrb2.0p25.f{step:03d}"
    file_path = os.path.join(grib_dir, f"cpofp_surface_{file_name}")
    url = (
        f"https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?file={file_name}"
        f"&lev_surface=on&var_CPOFP=on"
        f"&dir=%2Fgfs.{date_str}%2F{hour_str}%2Fatmos"
    )
    return download_grib(url, file_path)

def get_tmp_surface_grib(step):
    file_name = f"gfs.t{hour_str}z.pgrb2.0p25.f{step:03d}"
    file_path = os.path.join(grib_dir, f"tmp_surface_{file_name}")
    url = (
        f"https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?file={file_name}"
        f"&lev_2_m_above_ground=on&var_TMP=on"
        f"&dir=%2Fgfs.{date_str}%2F{hour_str}%2Fatmos"
    )
    return download_grib(url, file_path)

def get_wind_surface_grib(step):
    file_name = f"gfs.t{hour_str}z.pgrb2.0p25.f{step:03d}"
    file_path = os.path.join(grib_dir, f"wind_surface_{file_name}")
    url = (
        f"https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?file={file_name}"
        f"&lev_10_m_above_ground=on&var_UGRD=on&var_VGRD=on"
        f"&dir=%2Fgfs.{date_str}%2F{hour_str}%2Fatmos"
    )
    return download_grib(url, file_path)

def get_rh_surface_grib(step):
    file_name = f"gfs.t{hour_str}z.pgrb2.0p25.f{step:03d}"
    file_path = os.path.join(grib_dir, f"rh_surface_{file_name}")
    url = (
        f"https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?file={file_name}"
        f"&lev_2_m_above_ground=on&var_RH=on"
        f"&dir=%2Fgfs.{date_str}%2F{hour_str}%2Fatmos"
    )
    return download_grib(url, file_path)

def calculate_ice_risk(cfrzr, cpofp, tmp, rh, ugrd, vgrd):
    # Normalize temperature factor: favor near 0°C (e.g., Gaussian centered at 0°C)
    temp_factor = np.exp(-((tmp + 1)**2) / 4)  # Smooth transition around -1°C

    # Calculate wind speed and apply a factor: light wind favors icing
    wind_speed = np.sqrt(ugrd**2 + vgrd**2)
    wind_factor = np.clip((15 - wind_speed) / 15, 0, 1)  # Smooth transition for wind speeds

    # Relative humidity factor: smooth transition above 80%
    rh_factor = np.clip((rh - 80) / 20, 0, 1)  # Adjusted to require higher RH for significant contribution

    # Base risk calculation
    base_risk = cfrzr * cpofp

    # Thermodynamic and dynamic factors
    thermo_factor = temp_factor * rh_factor
    dynamic_factor = wind_factor

    # Final Ice Accretion Risk Index
    ice_risk = base_risk * thermo_factor * dynamic_factor

    # Define thresholds for different risk levels
    thresholds = {
        0.25: (cfrzr >= 0.1) & (cpofp >= 0.2) & (tmp >= -5) & (tmp <= 0) & (rh >= 85) & (wind_speed <= 10),
        0.50: (cfrzr >= 0.3) & (cpofp >= 0.4) & (tmp >= -3) & (tmp <= -0.5) & (rh >= 90) & (wind_speed <= 5),
        0.75: (cfrzr >= 0.5) & (cpofp >= 0.6) & (tmp >= -2) & (tmp <= -0.8) & (rh >= 95) & (wind_speed <= 3),
        1.00: (cfrzr >= 1) & (cpofp == 1) & (tmp >= -1.1) & (tmp <= -0.9) & (rh >= 99) & (wind_speed <= 1)
    }

    # Apply thresholds to adjust risk levels
    for risk_level, condition in thresholds.items():
        ice_risk[condition] = np.maximum(ice_risk[condition], risk_level)

    # Cap the risk at 100%
    ice_risk = np.clip(ice_risk, 0, 1.0)

    return ice_risk

def plot_ice_risk_surface(grib_path_frzr, grib_path_cpofp, grib_path_tmp, grib_path_wind, grib_path_rh, step):
    ds_frzr = xr.open_dataset(grib_path_frzr, engine="cfgrib", filter_by_keys={"stepType": "instant"})
    ds_cpofp = xr.open_dataset(grib_path_cpofp, engine="cfgrib", filter_by_keys={"stepType": "instant"})
    ds_tmp = xr.open_dataset(grib_path_tmp, engine="cfgrib", filter_by_keys={"stepType": "instant"})
    ds_wind = xr.open_dataset(grib_path_wind, engine="cfgrib", filter_by_keys={"stepType": "instant"})
    ds_rh = xr.open_dataset(grib_path_rh, engine="cfgrib", filter_by_keys={"stepType": "instant"})

    frzr = gaussian_filter(ds_frzr['cfrzr'].values, sigma=3)  # Freezing rain rate with smoothing
    cpofp = gaussian_filter(ds_cpofp['cpofp'].values, sigma=3)  # Probability of freezing precipitation with smoothing
    tmp = gaussian_filter(ds_tmp['t2m'].values - 273.15, sigma=3)  # Temperature (Celsius) with smoothing
    ugrd = ds_wind['u10'].values  # U-component of wind
    vgrd = ds_wind['v10'].values  # V-component of wind
    rh = gaussian_filter(ds_rh['r2'].values, sigma=3)  # Relative humidity with smoothing

    # Calculate Ice Accretion Risk Index
    ice_risk = calculate_ice_risk(frzr, cpofp, tmp, rh, ugrd, vgrd) * 100  # Convert to percentage

    lats = ds_frzr['latitude'].values
    lons = ds_frzr['longitude'].values

    Lon2d, Lat2d = np.meshgrid(lons, lats)

    fig = plt.figure(figsize=(10, 7), dpi=300, facecolor='white')
    ax = plt.axes([0.07, 0.13, 0.86, 0.85], projection=ccrs.PlateCarree(), facecolor='white')
    extent = [-130, -65, 20, 54]
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.LAND, facecolor='white')
    ax.add_feature(cfeature.OCEAN, facecolor='white')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.7)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.STATES, linewidth=0.6)

    # Define levels and colormap for Ice Accretion Risk Index (percentage)
    ice_risk_levels = np.linspace(0, 100, 21)  # More levels for smoother shading
    ice_risk_cmap = LinearSegmentedColormap.from_list("ice_risk_cmap", [
        '#ffffff', '#d0f0ff', '#a0e0ff', '#70d0ff', '#40c0ff', '#10b0ff', '#ffb0d0', '#ff80a0', '#ff5070', '#ff2040', "#9be216"
    ])

    contour_ice_risk = ax.contourf(
        Lon2d, Lat2d, ice_risk,
        levels=ice_risk_levels, cmap=ice_risk_cmap, extend='both', transform=ccrs.PlateCarree()
    )

    # Title block
    run_hour_map = {
        "00": 0,   # 00z run: f000 = 12:43 AM
        "06": 6,   # 06z run: f000 = 6:43 AM
        "12": 12,  # 12z run: f000 = 12:43 PM
        "18": 18   # 18z run: f000 = 6:43 PM
    }
    base_hour = run_hour_map.get(hour_str, 7)  # default to 7am if unknown
    base_time = datetime.strptime(date_str + f"{base_hour:02d}", "%Y%m%d%H") + timedelta(hours=1)  # Adjust for local time zone offset
    valid_time = base_time + timedelta(hours=step)
    hour_str_fmt = valid_time.strftime('%I%p').lstrip('0').lower()  # '08AM' -> '8am'
    day_of_week = valid_time.strftime('%A')  # Get the day of the week

    # Show local run time as the mapped local hour (e.g., 12z = 1pm, 18z = 7pm, etc)
    local_hour = run_hour_map.get(hour_str, 7)
    local_run_time = datetime.strptime(f"{date_str} {local_hour:02d}", "%Y%m%d %H") + timedelta(hours=1)
    local_run_time = local_run_time.strftime('%I%p').lstrip('0').lower()

    title = (
        f"GFS Model {valid_time.strftime('%y%m%d')} {local_run_time} {day_of_week} "
        f"Forecast Hour: {step} Run: {hour_str}z\nIce Accretion Risk Index (Percentage)"
    )
    plt.title(title, fontsize=12, fontweight='bold', y=1.03)

    # Color bar for Ice Accretion Risk Index (percentage)
    cbar_height = 0.012
    cbar_bottom = 0.12
    cax_tmp = fig.add_axes([0.10, cbar_bottom, 0.80, cbar_height])
    cbar = plt.colorbar(
        contour_ice_risk, cax=cax_tmp, orientation='horizontal',
        ticks=np.linspace(0, 100, 11), boundaries=ice_risk_levels
    )
    cbar.ax.set_xticklabels([f"{int(v)}%" for v in np.linspace(0, 100, 11)])
    cbar.set_label("Ice Accretion Risk Index (%)", fontsize=8, labelpad=2)
    cbar.ax.tick_params(labelsize=7, length=2)
    cbar.ax.set_facecolor('white')
    cbar.outline.set_edgecolor('black')

    png_path = os.path.join(png_dir, f"ice_risk_surface_{step:03d}.png")
    plt.savefig(png_path, bbox_inches='tight', pad_inches=0.3, dpi=300)
    plt.close(fig)
    print(f"Generated plot: {png_path}")
    return png_path

current_utc_time = datetime.utcnow() - timedelta(hours=6)
date_str = current_utc_time.strftime("%Y%m%d")
hour_str = str(current_utc_time.hour // 6 * 6).zfill(2)

for step in forecast_steps:
    grib_file_frzr = get_frzr_surface_grib(step)
    grib_file_cpofp = get_cpofp_surface_grib(step)
    grib_file_tmp = get_tmp_surface_grib(step)
    grib_file_wind = get_wind_surface_grib(step)
    grib_file_rh = get_rh_surface_grib(step)
    if grib_file_frzr and grib_file_cpofp and grib_file_tmp and grib_file_wind and grib_file_rh:
        plot_ice_risk_surface(grib_file_frzr, grib_file_cpofp, grib_file_tmp, grib_file_wind, grib_file_rh, step)
        gc.collect()
        time.sleep(1)

for f in os.listdir(grib_dir):
    file_path = os.path.join(grib_dir, f)
    if os.path.isfile(file_path):
        os.remove(file_path)

print("All tasks completed.")