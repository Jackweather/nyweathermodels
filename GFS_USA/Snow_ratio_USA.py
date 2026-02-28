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

grib_dir = os.path.join(BASE_DIR, "GFS_dzdt_700mb", "grib")
png_dir = os.path.join(BASE_DIR, "GFS_dzdt_700mb", "png")
os.makedirs(grib_dir, exist_ok=True)
os.makedirs(png_dir, exist_ok=True)

forecast_steps = list(range(6, 385, 6))

# Define levels and colormap for temperature at 700 mb
tmp_levels = np.linspace(-50, 50, 21)  # Temperature levels in °C
custom_cmap = LinearSegmentedColormap.from_list("tmp_cmap", [
    '#313695', '#4575b4', '#74add1', '#abd9e9', '#e0f3f8', '#ffffbf', '#fee090', '#fdae61', '#f46d43', '#d73027', '#a50026'
])
tmp_norm = BoundaryNorm(tmp_levels, custom_cmap.N)

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

def get_tmp_and_dzdt_700mb_grib(step):
    file_name = f"gfs.t{hour_str}z.pgrb2.1p00.f{step:03d}"
    file_path = os.path.join(grib_dir, f"tmp_dzdt_700mb_{file_name}")
    url = (
        f"https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_1p00.pl?file={file_name}"
        f"&lev_700_mb=on&var_TMP=on&var_DZDT=on"
        f"&dir=%2Fgfs.{date_str}%2F{hour_str}%2Fatmos"
    )
    return download_grib(url, file_path)

def get_tmp_and_dzdt_750mb_grib(step):
    file_name = f"gfs.t{hour_str}z.pgrb2.1p00.f{step:03d}"
    file_path = os.path.join(grib_dir, f"tmp_dzdt_750mb_{file_name}")
    url = (
        f"https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_1p00.pl?file={file_name}"
        f"&lev_750_mb=on&var_TMP=on&var_DZDT=on"
        f"&dir=%2Fgfs.{date_str}%2F{hour_str}%2Fatmos"
    )
    return download_grib(url, file_path)

def get_tmp_and_dzdt_800mb_grib(step):
    file_name = f"gfs.t{hour_str}z.pgrb2.1p00.f{step:03d}"
    file_path = os.path.join(grib_dir, f"tmp_dzdt_800mb_{file_name}")
    url = (
        f"https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_1p00.pl?file={file_name}"
        f"&lev_800_mb=on&var_TMP=on&var_DZDT=on"
        f"&dir=%2Fgfs.{date_str}%2F{hour_str}%2Fatmos"
    )
    return download_grib(url, file_path)

def get_tmp_and_dzdt_850mb_grib(step):
    file_name = f"gfs.t{hour_str}z.pgrb2.1p00.f{step:03d}"
    file_path = os.path.join(grib_dir, f"tmp_dzdt_850mb_{file_name}")
    url = (
        f"https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_1p00.pl?file={file_name}"
        f"&lev_850_mb=on&var_TMP=on&var_DZDT=on"
        f"&dir=%2Fgfs.{date_str}%2F{hour_str}%2Fatmos"
    )
    return download_grib(url, file_path)

def base_ratio_from_temp(dgz_temp):
    """
    Determine the base snow ratio based on the temperature in the DGZ (°C).

    Parameters:
        dgz_temp (float): Temperature in the DGZ (°C).

    Returns:
        int: Base snow ratio.
    """
    if dgz_temp <= -30:
        return 8  # Extremely cold, smaller crystals dominate
    elif -30 < dgz_temp <= -25:
        return 10  # Very cold, limited dendritic growth
    elif -25 < dgz_temp <= -22:
        return 13  # Favoring dendritic growth
    elif -22 < dgz_temp <= -18:
        return 16  # Near-optimal dendritic growth zone
    elif -18 < dgz_temp <= -15:
        return 19  # Peak dendritic growth zone
    elif -15 < dgz_temp <= -12:
        return 17  # Slightly warmer, still favorable for dendrites
    elif -12 < dgz_temp <= -8:
        return 14  # Warmer, mixed crystal types
    elif -8 < dgz_temp <= -5:
        return 12  # Transition to wetter snow
    elif -5 < dgz_temp <= -2:
        return 9   # Wet snow, lower ratios
    elif -2 < dgz_temp <= 0:
        return 7   # Very wet snow, compact flakes
    else:
        return 5   # Extremely wet snow, minimal ratios

def calculate_snow_ratio(dgz_temp, vvel_dgz):
    """
    Calculate snow ratio based on DGZ temperature and vertical velocity.

    Parameters:
        dgz_temp (float): Temperature in the DGZ (°C).
        vvel_dgz (float): Vertical velocity in the DGZ (cm/s).

    Returns:
        float: Snow ratio.
    """
    # Step 1: Base ratio from temperature
    ratio = base_ratio_from_temp(dgz_temp)

    # Step 2: Adjust for lift (vertical velocity)
    if vvel_dgz > 15:  # Strong lift threshold (cm/s)
        ratio += 3  # Boost for strong lift
    elif vvel_dgz > 5:  # Moderate lift threshold
        ratio += 1  # Small boost for lift
    elif vvel_dgz > 0:  # Weak lift
        ratio -= 1  # Slight penalty for weak lift

    # Step 3: Cap the ratio at realistic bounds
    return max(5, min(ratio, 24))

def plot_tmp_700mb(grib_path, step):
    ds = xr.open_dataset(grib_path, engine="cfgrib", filter_by_keys={"stepType": "instant"})

    tmp = gaussian_filter(ds['t'].values - 273.15, sigma=3)  # Convert from Kelvin to Celsius and smooth

    lats = ds['latitude'].values
    lons = ds['longitude'].values

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

    # Plot temperature at 700 mb
    contour_tmp = ax.contourf(
        Lon2d, Lat2d, tmp,
        levels=tmp_levels, cmap=custom_cmap, extend='both', transform=ccrs.PlateCarree()
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
        f"Forecast Hour: {step} Run: {hour_str}z\n700 mb Temperature (°C)"
    )
    plt.title(title, fontsize=12, fontweight='bold', y=1.03)

    # Color bar for temperature at 700 mb
    cbar_height = 0.012
    cbar_bottom = 0.12
    cax_tmp = fig.add_axes([0.10, cbar_bottom, 0.80, cbar_height])
    cbar = plt.colorbar(
        contour_tmp, cax=cax_tmp, orientation='horizontal',
        ticks=np.linspace(-50, 50, 11), boundaries=tmp_levels
    )
    cbar.ax.set_xticklabels([f"{int(v)}" for v in np.linspace(-50, 50, 11)])
    cbar.set_label("Temperature (°C)", fontsize=8, labelpad=2)
    cbar.ax.tick_params(labelsize=7, length=2)
    cbar.ax.set_facecolor('white')
    cbar.outline.set_edgecolor('black')

    png_path = os.path.join(png_dir, f"tmp_700mb_{step:03d}.png")
    plt.savefig(png_path, bbox_inches='tight', pad_inches=0.3, dpi=300)
    plt.close(fig)
    print(f"Generated plot: {png_path}")
    return png_path

def plot_tmp_700mb_and_750mb(grib_path_700mb, grib_path_750mb, step):
    ds_700mb = xr.open_dataset(grib_path_700mb, engine="cfgrib", filter_by_keys={"stepType": "instant"})
    ds_750mb = xr.open_dataset(grib_path_750mb, engine="cfgrib", filter_by_keys={"stepType": "instant"})

    tmp_700mb = gaussian_filter(ds_700mb['t'].values - 273.15, sigma=3)  # Convert from Kelvin to Celsius and smooth
    tmp_750mb = gaussian_filter(ds_750mb['t'].values - 273.15, sigma=3)  # Convert from Kelvin to Celsius and smooth

    lats = ds_700mb['latitude'].values
    lons = ds_700mb['longitude'].values

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

    # Plot temperature at 700 mb
    contour_tmp_700mb = ax.contour(
        Lon2d, Lat2d, tmp_700mb,
        levels=tmp_levels, colors='blue', linewidths=0.8, transform=ccrs.PlateCarree()
    )
    ax.clabel(contour_tmp_700mb, inline=True, fontsize=6, fmt="%.0f°C")

    # Plot temperature at 750 mb
    contour_tmp_750mb = ax.contour(
        Lon2d, Lat2d, tmp_750mb,
        levels=tmp_levels, colors='red', linewidths=0.8, transform=ccrs.PlateCarree()
    )
    ax.clabel(contour_tmp_750mb, inline=True, fontsize=6, fmt="%.0f°C")

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
        f"Forecast Hour: {step} Run: {hour_str}z\n700 mb and 750 mb Temperature (°C)"
    )
    plt.title(title, fontsize=12, fontweight='bold', y=1.03)

    png_path = os.path.join(png_dir, f"tmp_700mb_750mb_{step:03d}.png")
    plt.savefig(png_path, bbox_inches='tight', pad_inches=0.3, dpi=300)
    plt.close(fig)
    print(f"Generated plot: {png_path}")
    return png_path

def plot_tmp_700mb_750mb_800mb(grib_path_700mb, grib_path_750mb, grib_path_800mb, step):
    ds_700mb = xr.open_dataset(grib_path_700mb, engine="cfgrib", filter_by_keys={"stepType": "instant"})
    ds_750mb = xr.open_dataset(grib_path_750mb, engine="cfgrib", filter_by_keys={"stepType": "instant"})
    ds_800mb = xr.open_dataset(grib_path_800mb, engine="cfgrib", filter_by_keys={"stepType": "instant"})

    tmp_700mb = gaussian_filter(ds_700mb['t'].values - 273.15, sigma=3)  # Convert from Kelvin to Celsius and smooth
    tmp_750mb = gaussian_filter(ds_750mb['t'].values - 273.15, sigma=3)  # Convert from Kelvin to Celsius and smooth
    tmp_800mb = gaussian_filter(ds_800mb['t'].values - 273.15, sigma=3)  # Convert from Kelvin to Celsius and smooth

    lats = ds_700mb['latitude'].values
    lons = ds_700mb['longitude'].values

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

    # Plot temperature at 700 mb
    contour_tmp_700mb = ax.contour(
        Lon2d, Lat2d, tmp_700mb,
        levels=tmp_levels, colors='blue', linewidths=0.8, transform=ccrs.PlateCarree()
    )
    ax.clabel(contour_tmp_700mb, inline=True, fontsize=6, fmt="%.0f°C")

    # Plot temperature at 750 mb
    contour_tmp_750mb = ax.contour(
        Lon2d, Lat2d, tmp_750mb,
        levels=tmp_levels, colors='red', linewidths=0.8, transform=ccrs.PlateCarree()
    )
    ax.clabel(contour_tmp_750mb, inline=True, fontsize=6, fmt="%.0f°C")

    # Plot temperature at 800 mb
    contour_tmp_800mb = ax.contour(
        Lon2d, Lat2d, tmp_800mb,
        levels=tmp_levels, colors='green', linewidths=0.8, transform=ccrs.PlateCarree()
    )
    ax.clabel(contour_tmp_800mb, inline=True, fontsize=6, fmt="%.0f°C")

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
        f"Forecast Hour: {step} Run: {hour_str}z\n700 mb, 750 mb, and 800 mb Temperature (°C)"
    )
    plt.title(title, fontsize=12, fontweight='bold', y=1.03)

    png_path = os.path.join(png_dir, f"tmp_700mb_750mb_800mb_{step:03d}.png")
    plt.savefig(png_path, bbox_inches='tight', pad_inches=0.3, dpi=300)
    plt.close(fig)
    print(f"Generated plot: {png_path}")
    return png_path

def plot_tmp_700mb_750mb_800mb_850mb(grib_path_700mb, grib_path_750mb, grib_path_800mb, grib_path_850mb, grib_path_vvel_850mb, step):
    ds_700mb = xr.open_dataset(grib_path_700mb, engine="cfgrib", filter_by_keys={"stepType": "instant"})
    ds_750mb = xr.open_dataset(grib_path_750mb, engine="cfgrib", filter_by_keys={"stepType": "instant"})
    ds_800mb = xr.open_dataset(grib_path_800mb, engine="cfgrib", filter_by_keys={"stepType": "instant"})
    ds_850mb = xr.open_dataset(grib_path_850mb, engine="cfgrib", filter_by_keys={"stepType": "instant"})
    ds_vvel_850mb = xr.open_dataset(grib_path_vvel_850mb, engine="cfgrib", filter_by_keys={"stepType": "instant"})

    tmp_700mb = gaussian_filter(ds_700mb['t'].values - 273.15, sigma=3)  # Convert from Kelvin to Celsius and smooth
    tmp_750mb = gaussian_filter(ds_750mb['t'].values - 273.15, sigma=3)  # Convert from Kelvin to Celsius and smooth
    tmp_800mb = gaussian_filter(ds_800mb['t'].values - 273.15, sigma=3)  # Convert from Kelvin to Celsius and smooth
    tmp_850mb = gaussian_filter(ds_850mb['t'].values - 273.15, sigma=3)  # Convert from Kelvin to Celsius and smooth

    # Filter vertical velocity to only include positive values
    vvel_850mb = gaussian_filter(ds_vvel_850mb['w'].values, sigma=3)  # Smooth vertical velocity
    vvel_850mb = np.where(vvel_850mb > 0, vvel_850mb, np.nan)  # Set negative values to NaN

    lats = ds_700mb['latitude'].values
    lons = ds_700mb['longitude'].values

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

    # Plot temperature at 700 mb
    contour_tmp_700mb = ax.contour(
        Lon2d, Lat2d, tmp_700mb,
        levels=tmp_levels, colors='blue', linewidths=0.8, transform=ccrs.PlateCarree()
    )
    ax.clabel(contour_tmp_700mb, inline=True, fontsize=6, fmt="%.0f°C")

    # Plot temperature at 750 mb
    contour_tmp_750mb = ax.contour(
        Lon2d, Lat2d, tmp_750mb,
        levels=tmp_levels, colors='red', linewidths=0.8, transform=ccrs.PlateCarree()
    )
    ax.clabel(contour_tmp_750mb, inline=True, fontsize=6, fmt="%.0f°C")

    # Plot temperature at 800 mb
    contour_tmp_800mb = ax.contour(
        Lon2d, Lat2d, tmp_800mb,
        levels=tmp_levels, colors='green', linewidths=0.8, transform=ccrs.PlateCarree()
    )
    ax.clabel(contour_tmp_800mb, inline=True, fontsize=6, fmt="%.0f°C")

    # Plot temperature at 850 mb
    contour_tmp_850mb = ax.contour(
        Lon2d, Lat2d, tmp_850mb,
        levels=tmp_levels, colors='purple', linewidths=0.8, transform=ccrs.PlateCarree()
    )
    ax.clabel(contour_tmp_850mb, inline=True, fontsize=6, fmt="%.0f°C")

    # Plot vertical velocity at 850 mb
    contour_vvel_850mb = ax.contourf(
        Lon2d, Lat2d, vvel_850mb,
        levels=np.linspace(0, 1, 11), cmap="coolwarm", extend='both', transform=ccrs.PlateCarree()
    )

    # Add color bar for vertical velocity
    cbar = plt.colorbar(
        contour_vvel_850mb, ax=ax, orientation='horizontal', pad=0.05,
        aspect=50, label="Positive Vertical Velocity (m/s)"
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
        f"Forecast Hour: {step} Run: {hour_str}z\n700 mb, 750 mb, 800 mb, 850 mb Temperature and Positive Vertical Velocity"
    )
    plt.title(title, fontsize=12, fontweight='bold', y=1.03)

    png_path = os.path.join(png_dir, f"tmp_vvel_850mb_{step:03d}.png")
    plt.savefig(png_path, bbox_inches='tight', pad_inches=0.3, dpi=300)
    plt.close(fig)
    print(f"Generated plot: {png_path}")
    return png_path

def plot_snow_ratio(grib_paths, step):
    """
    Plot snow ratio map based on DGZ temperature and maximum upward vertical velocity.

    Parameters:
        grib_paths (dict): Dictionary containing paths to GRIB files for different levels.
        step (int): Forecast step.
    """
    # Open datasets for all levels
    ds_700mb = xr.open_dataset(grib_paths['700mb'], engine="cfgrib", filter_by_keys={"stepType": "instant"})
    ds_750mb = xr.open_dataset(grib_paths['750mb'], engine="cfgrib", filter_by_keys={"stepType": "instant"})
    ds_800mb = xr.open_dataset(grib_paths['800mb'], engine="cfgrib", filter_by_keys={"stepType": "instant"})
    ds_850mb = xr.open_dataset(grib_paths['850mb'], engine="cfgrib", filter_by_keys={"stepType": "instant"})
    # Assign vertical velocity data
    vvel_700mb = ds_700mb['wz'].values
    vvel_750mb = ds_750mb['wz'].values
    vvel_800mb = ds_800mb['wz'].values
    vvel_850mb = ds_850mb['wz'].values

    
    # Handle missing 'w' variable gracefully
    try:
        vvel_700mb = ds_700mb['wz'].values
        vvel_750mb = ds_750mb['wz'].values
        vvel_800mb = ds_800mb['wz'].values
        vvel_850mb = ds_850mb['wz'].values
    except KeyError as e:
        print("Vertical velocity data ('w') is missing in the dataset. Available variables:", list(ds_700mb.variables.keys()))
        raise e

    # Calculate DGZ temperature and maximum upward vertical velocity
    # Extract temperature and vertical velocity data
    tmp_700mb = ds_700mb['t'].values - 273.15  # Convert from Kelvin to Celsius
    tmp_750mb = ds_750mb['t'].values - 273.15
    tmp_800mb = ds_800mb['t'].values - 273.15
    tmp_850mb = ds_850mb['t'].values - 273.15

    max_vvel = np.maximum.reduce([vvel_700mb, vvel_750mb, vvel_800mb])

    # Calculate snow ratio
    snow_ratio = np.vectorize(calculate_snow_ratio)(tmp_700mb, max_vvel)

    lats = ds_700mb['latitude'].values
    lons = ds_700mb['longitude'].values
    Lon2d, Lat2d = np.meshgrid(lons, lats)

    # Define custom colormap for snow ratio
    snow_ratio_cmap = LinearSegmentedColormap.from_list("snow_ratio_cmap", [
        "#313695", "#4575b4", "#74add1", "#abd9e9", "#e0f3f8", "#ffffbf", "#fee090", "#fdae61", "#f46d43", "#d73027", "#a50026"
    ])

    # Plot snow ratio map
    fig = plt.figure(figsize=(10, 7), dpi=300, facecolor='white')
    ax = plt.axes([0.07, 0.13, 0.86, 0.85], projection=ccrs.PlateCarree(), facecolor='white')
    extent = [-130, -65, 20, 54]
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.LAND, facecolor='white')
    ax.add_feature(cfeature.OCEAN, facecolor='white')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.7)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.STATES, linewidth=0.6)

    contour = ax.contourf(
        Lon2d, Lat2d, snow_ratio,
        levels=np.linspace(4, 21, 18), cmap=snow_ratio_cmap, extend='both', transform=ccrs.PlateCarree()
    )

    cbar = plt.colorbar(
        contour, ax=ax, orientation='horizontal', pad=0.05,
        aspect=50, label="Snow Ratio (liquid:solid)"
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
        f"Forecast Hour: {step} Run: {hour_str}z\nSnow Ratio Map"
    )
    plt.title(title, fontsize=12, fontweight='bold', y=1.03)\

    png_path = os.path.join(png_dir, f"snow_ratio_{step:03d}.png")
    plt.savefig(png_path, bbox_inches='tight', pad_inches=0.3, dpi=300)
    plt.close(fig)
    print(f"Generated snow ratio plot: {png_path}")
    return png_path

current_utc_time = datetime.utcnow() - timedelta(hours=6)
date_str = current_utc_time.strftime("%Y%m%d")
hour_str = str(current_utc_time.hour // 6 * 6).zfill(2)

for step in forecast_steps:
    grib_paths = {
        '700mb': get_tmp_and_dzdt_700mb_grib(step),
        '750mb': get_tmp_and_dzdt_750mb_grib(step),
        '800mb': get_tmp_and_dzdt_800mb_grib(step),
        '850mb': get_tmp_and_dzdt_850mb_grib(step),
    }
    if all(grib_paths.values()):
        plot_snow_ratio(grib_paths, step)
        gc.collect()
        time.sleep(1)

# Remove other plotting functions and calls to ensure only snow ratio PNGs are generated.
