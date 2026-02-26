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

# Use a file lock so only one process downloads Cartopy data at a time.
lock = FileLock(os.path.join(os.getcwd(), 'cartopy.lock'))
with lock:
    shpreader = importlib.import_module('cartopy.io.shapereader')
    cfeature = importlib.import_module('cartopy.feature')

# Set base directory for HRRR output
BASE_DIR = '/var/data'

grib_dir = os.path.join(BASE_DIR, "GFS_tmp_2m", "grib")
png_dir = os.path.join(BASE_DIR, "GFS_tmp_2m", "png")
os.makedirs(grib_dir, exist_ok=True)
os.makedirs(png_dir, exist_ok=True)

forecast_steps = [0] + list(range(6, 385, 6))

# Define levels and colormap for temperature in Fahrenheit
temp_levels = [-20, 0, 10, 20, 32, 40, 50, 60, 70, 80, 90, 100]  # For colorbar use later, in Fahrenheit
custom_cmap = LinearSegmentedColormap.from_list(
    "temp_cmap",
    [
        "#08306b",  # very cold (dark blue)
        "#2171b5",  # cold (blue)
        "#6baed6",  # chilly (light blue)
        "#ffffff",  # freezing (white, 32F)
        "#ffffb2",  # cool (light yellow)
        "#fecc5c",  # mild (yellow)
        "#fd8d3c",  # warm (orange)
        "#f03b20",  # hot (red-orange)
        "#bd0026"   # very hot (dark red)
    ],
    N=256
)
temp_norm = BoundaryNorm(temp_levels, custom_cmap.N)

# Path to the county boundaries JSON file
COUNTY_JSON_PATH = r"/opt/render/project/src/counties.json"

# Load county boundaries
def load_county_boundaries(json_path):
    try:
        with open(json_path, 'r') as f:
            county_boundaries = json.load(f)
        print("Loaded county boundaries successfully.")
        return county_boundaries
    except Exception as e:
        print(f"Failed to load county boundaries: {e}")
        return None

# Load the county boundaries at the start
county_boundaries = load_county_boundaries(COUNTY_JSON_PATH)

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

def get_tmp_2m_grib(step):
    file_name = f"gfs.t{hour_str}z.pgrb2.0p25.f{step:03d}"
    file_path = os.path.join(grib_dir, f"tmp_2m_{file_name}")
    url = (
        f"https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?file={file_name}"
        f"&lev_2_m_above_ground=on&var_TMP=on"
        f"&dir=%2Fgfs.{date_str}%2F{hour_str}%2Fatmos"
    )
    return download_grib(url, file_path)

def plot_tmp_2m(grib_path, step):
    ds = xr.open_dataset(grib_path, engine="cfgrib")
    tmp_2m = (ds['t2m'].values - 273.15) * 9/5 + 32  # Convert from Kelvin to Fahrenheit
    lats = ds['latitude'].values
    lons = ds['longitude'].values
    Lon2d, Lat2d = np.meshgrid(lons, lats)

    fig = plt.figure(figsize=(10, 7), dpi=600, facecolor='white')
    ax = plt.axes([0.07, 0.13, 0.86, 0.85], projection=ccrs.PlateCarree(), facecolor='white')
    extent = [-130, -65, 20, 54]
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.LAND, facecolor='white')
    ax.add_feature(cfeature.OCEAN, facecolor='white')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.7)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.STATES, linewidth=0.6)

    contour = ax.contourf(
        Lon2d, Lat2d, tmp_2m,
        levels=temp_levels, cmap=custom_cmap, norm=temp_norm, transform=ccrs.PlateCarree()
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
        f"Forecast Hour: {step} Run: {hour_str}z\n2m Temperature (°F)"
    )
    plt.title(title, fontsize=12, fontweight='bold', y=1.03)

    # Color bar
    cbar_height = 0.012
    cbar_bottom = 0.12
    cax_tmp = fig.add_axes([0.10, cbar_bottom, 0.80, cbar_height])
    cbar = plt.colorbar(
        contour, cax=cax_tmp, orientation='horizontal',
        ticks=temp_levels, boundaries=temp_levels
    )
    cbar.ax.set_xticklabels([f"{v}°F" for v in temp_levels])
    cbar.set_label("2m Temperature (°F)", fontsize=8, labelpad=2)
    cbar.ax.tick_params(labelsize=7, length=2)
    cbar.ax.set_facecolor('white')
    cbar.outline.set_edgecolor('black')

    png_path = os.path.join(png_dir, f"tmp_2m_{step:03d}.png")
    plt.savefig(png_path, bbox_inches='tight', pad_inches=0.3, dpi=600)
    plt.close(fig)
    print(f"Generated plot: {png_path}")
    return png_path

current_utc_time = datetime.utcnow() - timedelta(hours=6)
date_str = current_utc_time.strftime("%Y%m%d")
hour_str = str(current_utc_time.hour // 6 * 6).zfill(2)

for step in forecast_steps:
    grib_file = get_tmp_2m_grib(step)
    if grib_file:
        plot_tmp_2m(grib_file, step)
        gc.collect()
        time.sleep(1)

for f in os.listdir(grib_dir):
    file_path = os.path.join(grib_dir, f)
    if os.path.isfile(file_path):
        os.remove(file_path)

print("All tasks completed.")