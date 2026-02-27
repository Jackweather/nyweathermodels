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

# Use a file lock so only one process downloads Cartopy data at a time.
lock = FileLock(os.path.join(os.getcwd(), 'cartopy.lock'))
with lock:
    shpreader = importlib.import_module('cartopy.io.shapereader')
    cfeature = importlib.import_module('cartopy.feature')

# Set base directory for HRRR output
BASE_DIR = '/var/data'

grib_dir = os.path.join(BASE_DIR, "GFS_absv_850mb", "grib")
png_dir = os.path.join(BASE_DIR, "GFS_absv_850mb", "png")
os.makedirs(grib_dir, exist_ok=True)
os.makedirs(png_dir, exist_ok=True)

forecast_steps = list(range(6, 385, 6))

# Define levels and colormap for positive vorticity advection (PVA)
pva_levels = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]  # Extended to 60
custom_cmap = LinearSegmentedColormap.from_list(
    "pva_cmap",
    [
        "#ffffff",  # 5 - white
        "#ffffb2",  # light yellow
        "#fed976",  # yellow-orange
        "#feb24c",  # orange
        "#fd8d3c",  # darker orange
        "#fc4e2a",  # red-orange
        "#e31a1c",  # red
        "#b10026"   # dark red
    ],
    N=256
)
vorticity_norm = BoundaryNorm(pva_levels, custom_cmap.N)

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

def get_absv_850mb_grib(step):
    file_name = f"gfs.t{hour_str}z.pgrb2.0p25.f{step:03d}"
    file_path = os.path.join(grib_dir, f"absv_850mb_{file_name}")
    url = (
        f"https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?file={file_name}"
        f"&lev_850_mb=on&var_ABSV=on"
        f"&dir=%2Fgfs.{date_str}%2F{hour_str}%2Fatmos"
    )
    return download_grib(url, file_path)

def get_mslp_grib(step):
    file_name = f"gfs.t{hour_str}z.pgrb2.0p25.f{step:03d}"
    file_path = os.path.join(grib_dir, f"mslp_{file_name}")
    url = (
        f"https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?file={file_name}"
        f"&lev_mean_sea_level=on&var_MSLET=on"
        f"&dir=%2Fgfs.{date_str}%2F{hour_str}%2Fatmos"
    )
    return download_grib(url, file_path)

def plot_absv_850mb(grib_path, step):
    ds = xr.open_dataset(grib_path, engine="cfgrib")
    absv_850mb = ds['absv'].values * 1e5  # Convert to 10^-5 s^-1

    # Filter for only positive vorticity values
    absv_850mb = np.maximum(absv_850mb, 0)

    # Apply Gaussian smoothing to the data
    absv_850mb = gaussian_filter(absv_850mb, sigma=1)

    # Adjust the sub-sampling to reduce smoothing while maintaining faster plotting
    lats = ds['latitude'].values[::7]  # Take every 7th point
    lons = ds['longitude'].values[::7]  # Take every 7th point
    absv_850mb = absv_850mb[::7, ::7]  # Sub-sample the data grid less aggressively

    Lon2d, Lat2d = np.meshgrid(lons, lats)

    fig = plt.figure(figsize=(10, 7), dpi=300, facecolor='white')  # Reduced DPI for faster rendering
    ax = plt.axes([0.07, 0.13, 0.86, 0.85], projection=ccrs.PlateCarree(), facecolor='white')
    extent = [-130, -65, 20, 54]
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.LAND, facecolor='white')
    ax.add_feature(cfeature.OCEAN, facecolor='white')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.7)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.STATES, linewidth=0.6)

    contour = ax.contourf(
        Lon2d, Lat2d, absv_850mb,
        levels=pva_levels, cmap=custom_cmap, norm=vorticity_norm, transform=ccrs.PlateCarree()
    )

    # Map GFS run hour to local base time for f000
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

    # Title block
    title = (
        f"GFS Model {valid_time.strftime('%y%m%d')} {local_run_time} {day_of_week} "
        f"Forecast Hour: {step} Run: {hour_str}z\n850mb Positive Vorticity (10^-5 s^-1)"
    )
    plt.title(title, fontsize=12, fontweight='bold', y=1.03)

    # Color bar
    cbar_height = 0.012
    cbar_bottom = 0.12
    cax_tmp = fig.add_axes([0.10, cbar_bottom, 0.80, cbar_height])
    cbar = plt.colorbar(
        contour, cax=cax_tmp, orientation='horizontal',
        ticks=pva_levels, boundaries=pva_levels
    )
    cbar.ax.set_xticklabels([f"{v}" for v in pva_levels])
    cbar.set_label("850mb Positive Vorticity (10^-5 s^-1)", fontsize=8, labelpad=2)
    cbar.ax.tick_params(labelsize=7, length=2)
    cbar.ax.set_facecolor('white')
    cbar.outline.set_edgecolor('black')

    png_path = os.path.join(png_dir, f"absv_850mb_{step:03d}.png")
    plt.savefig(png_path, bbox_inches='tight', pad_inches=0.3, dpi=300)  # Reduced DPI for faster rendering
    plt.close(fig)
    print(f"Generated plot: {png_path}")
    return png_path

def plot_absv_850mb_with_mslp(grib_path, mslp_path, step):
    ds = xr.open_dataset(grib_path, engine="cfgrib")
    absv_850mb = ds['absv'].values * 1e5  # Convert to 10^-5 s^-1

    # Filter for only positive vorticity values
    absv_850mb = np.maximum(absv_850mb, 0)

    # Apply Gaussian smoothing to the data
    absv_850mb = gaussian_filter(absv_850mb, sigma=1)

    # Adjust the sub-sampling to reduce smoothing while maintaining faster plotting
    lats = ds['latitude'].values[::7]  # Take every 7th point
    lons = ds['longitude'].values[::7]  # Take every 7th point
    absv_850mb = absv_850mb[::7, ::7]  # Sub-sample the data grid less aggressively

    Lon2d, Lat2d = np.meshgrid(lons, lats)

    # Load MSLP data
    ds_mslp = xr.open_dataset(mslp_path, engine="cfgrib")
    mslp = ds_mslp['mslet'].values / 100.0  # Convert Pa to hPa
    mslp = mslp[::7, ::7]  # Sub-sample the data grid

    fig = plt.figure(figsize=(10, 7), dpi=300, facecolor='white')  # Reduced DPI for faster rendering
    ax = plt.axes([0.07, 0.13, 0.86, 0.85], projection=ccrs.PlateCarree(), facecolor='white')
    extent = [-130, -65, 20, 54]
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.LAND, facecolor='white')
    ax.add_feature(cfeature.OCEAN, facecolor='white')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.7)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.STATES, linewidth=0.6)

    contour = ax.contourf(
        Lon2d, Lat2d, absv_850mb,
        levels=pva_levels, cmap=custom_cmap, norm=vorticity_norm, transform=ccrs.PlateCarree()
    )

    # Add MSLP contours
    mslp_contour = ax.contour(
        Lon2d, Lat2d, mslp,
        levels=np.arange(960, 1050, 4), colors='black', linewidths=0.5, transform=ccrs.PlateCarree()
    )
    ax.clabel(mslp_contour, fmt='%d', fontsize=6, inline=1)

    # Map GFS run hour to local base time for f000
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

    # Title block
    title = (
        f"GFS Model {valid_time.strftime('%y%m%d')} {local_run_time} {day_of_week} "
        f"Forecast Hour: {step} Run: {hour_str}z\n850mb Positive Vorticity and MSLP"
    )
    plt.title(title, fontsize=12, fontweight='bold', y=1.03)

    # Color bar
    cbar_height = 0.012
    cbar_bottom = 0.12
    cax_tmp = fig.add_axes([0.10, cbar_bottom, 0.80, cbar_height])
    cbar = plt.colorbar(
        contour, cax=cax_tmp, orientation='horizontal',
        ticks=pva_levels, boundaries=pva_levels
    )
    cbar.ax.set_xticklabels([f"{v}" for v in pva_levels])
    cbar.set_label("850mb Positive Vorticity (10^-5 s^-1)", fontsize=8, labelpad=2)
    cbar.ax.tick_params(labelsize=7, length=2)
    cbar.ax.set_facecolor('white')
    cbar.outline.set_edgecolor('black')

    png_path = os.path.join(png_dir, f"absv_850mb_mslp_{step:03d}.png")
    plt.savefig(png_path, bbox_inches='tight', pad_inches=0.3, dpi=300)  # Reduced DPI for faster rendering
    plt.close(fig)
    print(f"Generated plot: {png_path}")
    return png_path

current_utc_time = datetime.utcnow() - timedelta(hours=6)
date_str = current_utc_time.strftime("%Y%m%d")
hour_str = str(current_utc_time.hour // 6 * 6).zfill(2)

for step in forecast_steps:
    grib_file = get_absv_850mb_grib(step)
    mslp_file = get_mslp_grib(step)
    if grib_file and mslp_file:
        plot_absv_850mb_with_mslp(grib_file, mslp_file, step)
        gc.collect()
        time.sleep(1)

for f in os.listdir(grib_dir):
    file_path = os.path.join(grib_dir, f)
    if os.path.isfile(file_path):
        os.remove(file_path)

print("All tasks completed.")