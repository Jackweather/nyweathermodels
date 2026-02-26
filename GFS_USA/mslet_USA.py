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
import scipy.ndimage as ndimage
import scipy.interpolate as interp  # <-- add this import
import time
import gc
from filelock import FileLock
import cartopy
import importlib
import geopandas as gpd  # Import geopandas for handling GeoJSON files
import shutil

# Use a file lock so only one process downloads Cartopy data at a time.
# Other processes will wait until the lock is released.
lock = FileLock(os.path.join(os.getcwd(), 'cartopy.lock'))  # Updated lock location
with lock:
    # import modules while holding the lock
    shpreader = importlib.import_module('cartopy.io.shapereader')
    cfeature = importlib.import_module('cartopy.feature')

# Set base directory for HRRR output
BASE_DIR = '/var/data'

# Output directories (use new region folder)
grib_dir = os.path.join(BASE_DIR, "GSF_mslp_prate_csnow", "grib")
png_dir = os.path.join(BASE_DIR, "GSF_mslp_prate_csnow", "png")
os.makedirs(grib_dir, exist_ok=True)

# Remove entire png folder if it exists (ensures it's truly cleared), then recreate it
if os.path.isdir(png_dir):
    try:
        shutil.rmtree(png_dir)
        print(f"Removed existing png directory: {png_dir}")
    except Exception as e:
        print(f"Failed to remove png directory {png_dir}: {e}")
os.makedirs(png_dir, exist_ok=True)

# File to track completed forecast steps for the current run
processed_steps_file = os.path.join(BASE_DIR, "GSF_mslp_prate_csnow", "processed_steps.txt")

# Clear the processed steps file at the start of a new run
if os.path.exists(processed_steps_file):
    os.remove(processed_steps_file)
# GFS 0.25-degree URL and variables
base_url_0p25 = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"
variable_mslp = "MSLET"
variable_prate = "PRATE"
variable_csnow = "CSNOW"
variable_cfrzr = "CFRZR"  # Freezing Rain Rate
variable_cicep = "CICEP"  # Probability of Ice Pellets

# Current UTC time minus 6 hours (nearest available GFS cycle)
current_utc_time = datetime.utcnow() - timedelta(hours=6)
date_str = current_utc_time.strftime("%Y%m%d")
hour_str = str(current_utc_time.hour // 6 * 6).zfill(2)  # nearest 6-hour slot

# Levels and colormaps
mslp_levels = np.arange(960, 1050+2, 2)
mslp_cmap = LinearSegmentedColormap.from_list(
    "mslp_cmap",
    [
        "#08306b", "#2171b5", "#6baed6", "#b3cde3", "#ffffff",
        "#ffffb2", "#fecc5c", "#fd8d3c", "#f03b20", "#bd0026"
    ],
    N=256
)
prate_levels = [0.1, 0.25, 0.5, 0.75, 1.5, 2, 2.5, 3, 4, 6, 10, 16, 24]
prate_colors = [
    "#b6ffb6",  # very light green
    "#54f354",  # light green
    "#19a319",  # medium green
    "#016601",  # dark green
    "#c9c938",  # bright yellow
    "#f5f825",  # dark yellow
    "#ffd700",  # gold
    "#ffa500",  # orange
    "#ff7f50",  # coral
    "#ff4500",  # orange-red
    "#ff1493",  # deep pink
    "#9400d3"   # dark violet
]
prate_cmap = LinearSegmentedColormap.from_list("prate_custom", prate_colors, N=len(prate_colors))
prate_norm = BoundaryNorm(prate_levels, prate_cmap.N)
snow_levels = [0.10, 0.25, 0.5, 1, 2, 4, 8, 16]
snow_colors = [
    "#e3f2fd",  # 0.10 very light blue
    "#bbdefb",  # 0.25 light blue
    "#90caf9",  # 0.5 blue
    "#42a5f5",  # 1 medium blue
    "#1e88e5",  # 2 deeper blue
    "#1565c0",  # 4 dark blue
    "#0d47a1",  # 8 very dark blue
    "#002171",  # 16 almost navy
]
snow_cmap = LinearSegmentedColormap.from_list("snow_cbar", snow_colors, N=len(snow_colors))
snow_norm = BoundaryNorm(snow_levels, snow_cmap.N)
cfrzr_levels = [0.10, 0.25, 0.5, 1, 2, 4, 8, 16]
cfrzr_colors = [
    "#fce4ec",  # very light pink
    "#f8bbd0",  # light pink
    "#f48fb1",  # medium pink
    "#ec407a",  # dark pink
    "#d81b60",  # deep pink
    "#880e4f",  # very dark pink
    "#560027",  # almost maroon
]
cfrzr_cmap = LinearSegmentedColormap.from_list("cfrzr_cbar", cfrzr_colors, N=len(cfrzr_colors))
cfrzr_norm = BoundaryNorm(cfrzr_levels, cfrzr_cmap.N)
cicep_levels = [0.10, 0.25, 0.5, 1, 2, 4, 8, 16]
cicep_colors = [
    "#f3e5f5",  # very light purple
    "#e1bee7",  # light purple
    "#ce93d8",  # medium purple
    "#ba68c8",  # dark purple
    "#9c27b0",  # deep purple
    "#6a1b9a",  # very dark purple
    "#4a148c",  # almost violet
]
cicep_cmap = LinearSegmentedColormap.from_list("cicep_cbar", cicep_colors, N=len(cicep_colors))
cicep_norm = BoundaryNorm(cicep_levels, cicep_cmap.N)

# Forecast steps
forecast_steps = [0] + list(range(6, 385, 6))


# Download functions
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

def get_mslp_grib(step):
    if step == 0:
        file_name = f"gfs.t{hour_str}z.pgrb2.0p25.f000"
    else:
        file_name = f"gfs.t{hour_str}z.pgrb2.0p25.f{step:03d}"
    file_path = os.path.join(grib_dir, f"mslp_{file_name}")
    url = (
        f"{base_url_0p25}?file={file_name}"
        f"&lev_mean_sea_level=on&var_{variable_mslp}=on"
        f"&subregion=&leftlon=220&rightlon=300&toplat=55&bottomlat=20"
        f"&dir=%2Fgfs.{date_str}%2F{hour_str}%2Fatmos"
    )
    return download_grib(url, file_path)

def get_prate_grib(step):
    if step == 0:
        file_name = f"gfs.t{hour_str}z.pgrb2.0p25.f000"
    else:
        file_name = f"gfs.t{hour_str}z.pgrb2.0p25.f{step:03d}"
    file_path = os.path.join(grib_dir, f"prate_{file_name}")
    url = (
        f"{base_url_0p25}?file={file_name}"
        f"&lev_surface=on&var_{variable_prate}=on"
        f"&subregion=&leftlon=220&rightlon=300&toplat=55&bottomlat=20"
        f"&dir=%2Fgfs.{date_str}%2F{hour_str}%2Fatmos"
    )
    return download_grib(url, file_path)

def get_csnow_grib(step):
    if step == 0:
        file_name = f"gfs.t{hour_str}z.pgrb2.0p25.f000"
    else:
        file_name = f"gfs.t{hour_str}z.pgrb2.0p25.f{step:03d}"
    file_path = os.path.join(grib_dir, f"csnow_{file_name}")
    url = (
        f"{base_url_0p25}?file={file_name}"
        f"&lev_surface=on&var_{variable_csnow}=on"
        f"&subregion=&leftlon=220&rightlon=300&toplat=55&bottomlat=20"
        f"&dir=%2Fgfs.{date_str}%2F{hour_str}%2Fatmos"
    )
    return download_grib(url, file_path)

def get_cfrzr_grib(step):
    if step == 0:
        file_name = f"gfs.t{hour_str}z.pgrb2.0p25.f000"
    else:
        file_name = f"gfs.t{hour_str}z.pgrb2.0p25.f{step:03d}"
    file_path = os.path.join(grib_dir, f"cfrzr_{file_name}")
    url = (
        f"{base_url_0p25}?file={file_name}"
        f"&lev_surface=on&var_{variable_cfrzr}=on"
        f"&subregion=&leftlon=220&rightlon=300&toplat=55&bottomlat=20"
        f"&dir=%2Fgfs.{date_str}%2F{hour_str}%2Fatmos"
    )
    return download_grib(url, file_path)

def get_cicep_grib(step):
    if step == 0:
        file_name = f"gfs.t{hour_str}z.pgrb2.0p25.f000"
    else:
        file_name = f"gfs.t{hour_str}z.pgrb2.0p25.f{step:03d}"
    file_path = os.path.join(grib_dir, f"cicep_{file_name}")
    url = (
        f"{base_url_0p25}?file={file_name}"
        f"&lev_surface=on&var_{variable_cicep}=on"
        f"&subregion=&leftlon=220&rightlon=300&toplat=55&bottomlat=20"
        f"&dir=%2Fgfs.{date_str}%2F{hour_str}%2Fatmos"
    )
    return download_grib(url, file_path)

# Path to the county boundaries JSON file
COUNTY_JSON_PATH = r"counties.json"

# Load county boundaries
def load_county_boundaries(json_path):
    try:
        counties = gpd.read_file(json_path)
        print(f"Loaded {len(counties)} counties from {json_path}")
        return counties
    except Exception as e:
        print(f"Failed to load county boundaries: {e}")
        return None

# Plotting function
def plot_combined(mslp_path, prate_path, step, csnow_path=None, cfrzr_path=None, cicep_path=None):
    try:
        ds_mslp = xr.open_dataset(mslp_path, engine="cfgrib")
        ds_prate = xr.open_dataset(prate_path, engine="cfgrib", filter_by_keys={"stepType": "instant"})
        ds_csnow = None
        ds_cfrzr = None
        ds_cicep = None
        if csnow_path and os.path.exists(csnow_path):
            try:
                ds_csnow = xr.open_dataset(csnow_path, engine="cfgrib", filter_by_keys={"stepType": "instant"})
            except Exception as e:
                print(f"Error opening CSNOW dataset: {e}")
                ds_csnow = None
        if cfrzr_path and os.path.exists(cfrzr_path):
            try:
                ds_cfrzr = xr.open_dataset(cfrzr_path, engine="cfgrib", filter_by_keys={"stepType": "instant"})
            except Exception as e:
                print(f"Error opening CFRZR dataset: {e}")
                ds_cfrzr = None
        if cicep_path and os.path.exists(cicep_path):
            try:
                ds_cicep = xr.open_dataset(cicep_path, engine="cfgrib", filter_by_keys={"stepType": "instant"})
            except Exception as e:
                print(f"Error opening CICEP dataset: {e}")
                ds_cicep = None
    except Exception as e:
        print(f"Error opening dataset: {e}")
        return None

    mslp = ds_mslp['mslet'].values / 100.0  # Pa to hPa
    prate = ds_prate['prate'].values * 3600  # mm/s to mm/hr

    lats = ds_mslp['latitude'].values
    lons = ds_mslp['longitude'].values
    lons_plot = np.where(lons > 180, lons - 360, lons)
    if lats.ndim == 1 and lons.ndim == 1:
        Lon2d, Lat2d = np.meshgrid(lons_plot, lats)
        mslp2d = mslp.squeeze()
        prate2d = prate.squeeze()
    else:
        Lon2d, Lat2d = lons_plot, lats
        mslp2d = mslp.squeeze()
        prate2d = prate.squeeze()

    data2d = mslp2d  # <-- Move this line here, before snow mask logic

    # --- Snow mask and snow rate ---
    snow_mask = None
    snow_rate2d = None
    if ds_csnow is not None and "csnow" in ds_csnow:
        csnow = ds_csnow['csnow'].values * 3600  # mm/s to mm/hr
        if csnow.shape == prate2d.shape:
            snow_mask = (csnow > 0)
            snow_rate2d = np.where(snow_mask, prate2d, np.nan)
        else:
            try:
                csnow2d = csnow.squeeze()
                snow_mask = (csnow2d > 0)
                snow_rate2d = np.where(snow_mask, prate2d, np.nan)
            except Exception:
                snow_mask = None
                snow_rate2d = None

    # --- Freezing Rain mask and rate ---
    cfrzr_mask = None
    cfrzr_rate2d = None
    if ds_cfrzr is not None and "cfrzr" in ds_cfrzr:
        cfrzr = ds_cfrzr['cfrzr'].values * 3600  # mm/s to mm/hr
        if cfrzr.shape == prate2d.shape:
            cfrzr_mask = (cfrzr > 0)
            cfrzr_rate2d = np.where(cfrzr_mask, prate2d, np.nan)
        else:
            try:
                cfrzr2d = cfrzr.squeeze()
                cfrzr_mask = (cfrzr2d > 0)
                cfrzr_rate2d = np.where(cfrzr_mask, prate2d, np.nan)
            except Exception:
                cfrzr_mask = None
                cfrzr_rate2d = None

    # --- Ice Pellets mask and rate ---
    cicep_mask = None
    cicep_rate2d = None
    if ds_cicep is not None and "cicep" in ds_cicep:
        cicep = ds_cicep['cicep'].values * 3600  # mm/s to mm/hr
        if cicep.shape == prate2d.shape:
            cicep_mask = (cicep > 0)
            cicep_rate2d = np.where(cicep_mask, prate2d, np.nan)
        else:
            try:
                cicep2d = cicep.squeeze()
                cicep_mask = (cicep2d > 0)
                cicep_rate2d = np.where(cicep_mask, prate2d, np.nan)
            except Exception:
                cicep_mask = None
                cicep_rate2d = None

    # Remove PRATE masking so it fills all areas
    prate2d_base = prate2d

    # --- Begin custom basemap integration ---
    fig = plt.figure(figsize=(10, 7), dpi=600, facecolor='white')
    # Make axes take even more vertical space, colorbars right below
    ax = plt.axes([0.07, 0.13, 0.86, 0.85], projection=ccrs.PlateCarree(), facecolor='white')
    extent = [-130, -65, 20, 54]
    margin = 1.0  # degrees margin to avoid plotting on the very edge

    # --- Title block ---
    # Map GFS run hour to local base time for f000
    run_hour_map = {
        "00": 1,   # 00z run: f000 = 1am
        "06": 7,   # 06z run: f000 = 7am
        "12": 13,  # 12z run: f000 = 1pm
        "18": 19   # 18z run: f000 = 7pm
    }
    base_hour = run_hour_map.get(hour_str, 7)  # default to 7am if unknown
    base_time = datetime.strptime(date_str + f"{base_hour:02d}", "%Y%m%d%H")
    valid_time = base_time + timedelta(hours=step)
    hour_str_fmt = valid_time.strftime('%I%p').lstrip('0').lower()  # '08AM' -> '8am'
    day_of_week = valid_time.strftime('%A')  # Get the day of the week
    # Show local run time as the mapped local hour (e.g., 12z = 1pm, 18z = 7pm, etc)
    local_hour = run_hour_map.get(hour_str, 7)
    local_run_time = datetime.strptime(f"{date_str} {local_hour:02d}", "%Y%m%d %H").strftime('%I%p').lstrip('0').lower()
    title = (
        f"GFS Model {valid_time.strftime('%y%m%d')} {local_run_time} {day_of_week} "
        f"Forecast Hour: {step} Run: {hour_str}z\nPrecipitation Rate & Mean Sea Level Pressure"
    )
    # Add title above plot
    plt.title(title, fontsize=12, fontweight='bold', y=1.03)

    def in_extent(lon, lat):
        return (extent[0] + margin <= lon <= extent[1] - margin) and (extent[2] + margin <= lat <= extent[3] - margin)

    # Define the map extent
    extent = [-130, -65, 20, 54]  # [west, east, south, north]

    # Function to mask data outside the extent
    def mask_outside_extent(data, lons, lats, extent):
        mask = (
            (lons >= extent[0]) & (lons <= extent[1]) &
            (lats >= extent[2]) & (lats <= extent[3])
        )
        return np.where(mask, data, np.nan)

    # Apply the mask to the data arrays
    prate2d_base = mask_outside_extent(prate2d, Lon2d, Lat2d, extent)
    if snow_rate2d is not None:
        snow_rate2d = mask_outside_extent(snow_rate2d, Lon2d, Lat2d, extent)
    if cfrzr_rate2d is not None:
        cfrzr_rate2d = mask_outside_extent(cfrzr_rate2d, Lon2d, Lat2d, extent)
    if cicep_rate2d is not None:
        cicep_rate2d = mask_outside_extent(cicep_rate2d, Lon2d, Lat2d, extent)
    mslp2d = mask_outside_extent(mslp2d, Lon2d, Lat2d, extent)

    # Set the map extent
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    # Base map
    ax.add_feature(cfeature.LAND, facecolor='white')
    ax.add_feature(cfeature.OCEAN, facecolor='white')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.7)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.STATES, linewidth=0.6)
    ax.add_feature(cfeature.RIVERS, linewidth=0.4, edgecolor='blue')
    ax.add_feature(cfeature.LAKES, facecolor='lightblue', edgecolor='blue', linewidth=0.3)
    # --- End custom basemap integration ---

    # Plot PRATE everywhere as the base layer
    mesh = ax.contourf(
        Lon2d, Lat2d, prate2d_base,
        levels=prate_levels,
        cmap=prate_cmap,
        norm=prate_norm,
        extend='max',
        transform=ccrs.PlateCarree(),
        alpha=0.7,
        zorder=2,
        edgecolors='white'
    )

    # Overlay snow rate only where snow is present
    if snow_rate2d is not None:
        snow_mesh = ax.contourf(
            Lon2d, Lat2d, snow_rate2d,
            levels=snow_levels,
            cmap=snow_cmap,
            norm=snow_norm,
            extend='max',
            transform=ccrs.PlateCarree(),
            alpha=0.85,
            zorder=3
        )

    # Overlay freezing rain rate only where freezing rain is present
    if cfrzr_rate2d is not None:
        cfrzr_mesh = ax.contourf(
            Lon2d, Lat2d, cfrzr_rate2d,
            levels=cfrzr_levels,
            cmap=cfrzr_cmap,
            norm=cfrzr_norm,
            extend='max',
            transform=ccrs.PlateCarree(),
            alpha=0.85,
            zorder=4
        )

    # Overlay ice pellets rate only where present
    if cicep_rate2d is not None:
        cicep_mesh = ax.contourf(
            Lon2d, Lat2d, cicep_rate2d,
            levels=cicep_levels,
            cmap=cicep_cmap,
            norm=cicep_norm,
            extend='max',
            transform=ccrs.PlateCarree(),
            alpha=0.85,
            zorder=5
        )

    # --- Place all colorbars just below the map, touching the plot ---
    cbar_height = 0.012
    cbar_bottom = 0.12  # just below the axes bottom (which is at 0.13)
    cax_prate = fig.add_axes([0.10, cbar_bottom, 0.18, cbar_height])  # leftmost colorbar
    cax_snow = fig.add_axes([0.33, cbar_bottom, 0.18, cbar_height])  # second colorbar
    cax_cfrzr = fig.add_axes([0.56, cbar_bottom, 0.18, cbar_height])  # third colorbar
    cax_cicep = fig.add_axes([0.79, cbar_bottom, 0.18, cbar_height])  # rightmost colorbar

    # PRATE colorbar
    cbar = plt.colorbar(
        mesh, cax=cax_prate, orientation='horizontal',
        ticks=prate_levels[::2], boundaries=prate_levels  # Use fewer ticks for clarity
    )
    prate_tick_labels = [f"{v:g}" for v in prate_levels[::2]]  # Remove unnecessary zeros
    cbar.ax.set_xticklabels(prate_tick_labels)
    cbar.set_label("Precipitation Rate (mm/hr)", fontsize=8, labelpad=2)
    cbar.ax.tick_params(labelsize=7, length=2)
    cbar.ax.set_facecolor('white')
    cbar.outline.set_edgecolor('black')

    # SNOW colorbar
    if snow_rate2d is not None:
        cbar_snow = plt.colorbar(
            snow_mesh, cax=cax_snow, orientation='horizontal',
            ticks=snow_levels, boundaries=snow_levels  # Include all levels up to 16
        )
        snow_tick_labels = [f"{v:g}" for v in snow_levels]  # Remove unnecessary zeros
        cbar_snow.ax.set_xticklabels(snow_tick_labels)
        cbar_snow.set_label("Snow Rate (mm/hr)", fontsize=8, labelpad=2)
        cbar_snow.ax.tick_params(labelsize=7, length=2)
        cbar_snow.ax.set_facecolor('white')
        cbar_snow.outline.set_edgecolor('black')

    # CFRZR colorbar
    if cfrzr_rate2d is not None:
        cbar_cfrzr = plt.colorbar(
            cfrzr_mesh, cax=cax_cfrzr, orientation='horizontal',
            ticks=cfrzr_levels, boundaries=cfrzr_levels  # Include all levels up to 16
        )
        cfrzr_tick_labels = [f"{v:g}" for v in cfrzr_levels]  # Remove unnecessary zeros
        cbar_cfrzr.ax.set_xticklabels(cfrzr_tick_labels)
        cbar_cfrzr.set_label("Freezing Rain Rate (mm/hr)", fontsize=8, labelpad=2)
        cbar_cfrzr.ax.tick_params(labelsize=7, length=2)
        cbar_cfrzr.ax.set_facecolor('white')
        cbar_cfrzr.outline.set_edgecolor('black')

    # CICEP colorbar
    if cicep_rate2d is not None:
        cbar_cicep = plt.colorbar(
            cicep_mesh, cax=cax_cicep, orientation='horizontal',
            ticks=cicep_levels, boundaries=cicep_levels  # Include all levels up to 16
        )
        cicep_tick_labels = [f"{v:g}" for v in cicep_levels]  # Remove unnecessary zeros
        cbar_cicep.ax.set_xticklabels(cicep_tick_labels)
        cbar_cicep.set_label("Ice Pellets Rate (mm/hr)", fontsize=8, labelpad=2)
        cbar_cicep.ax.tick_params(labelsize=7, length=2)
        cbar_cicep.ax.set_facecolor('white')
        cbar_cicep.outline.set_edgecolor('black')

    # --- MSLP plotting ---
    cs = ax.contour(
        Lon2d, Lat2d, mslp2d,
        levels=mslp_levels,
        colors='black',
        linewidths=0.7,  # thinner lines
        transform=ccrs.PlateCarree()
    )
    ax.clabel(cs, fmt='%d', fontsize=4, colors='black', inline=True)

    # --- Highs and Lows Detection ---
    mask = (
        (Lon2d >= extent[0]) & (Lon2d <= extent[1]) &
        (Lat2d >= extent[2]) & (Lat2d <= extent[3])
    )
    data_masked = np.where(mask, mslp2d, np.nan)

    # Detect Lows
    min_filt = ndimage.minimum_filter(data_masked, size=25, mode='constant', cval=np.nan)
    lows = (data_masked == min_filt) & ~np.isnan(data_masked) & (data_masked <= 1006)
    low_y, low_x = np.where(lows)
    lows_plotted = 0
    max_lows = 4
    min_low_distance = 60  # minimum grid points between plotted lows
    plotted_low_points = []
    for y, x in zip(low_y, low_x):
        if lows_plotted >= max_lows:
            break
        lon, lat = Lon2d[y, x], Lat2d[y, x]
        # Skip lows near the edge of the map
        if not ((extent[0] + margin <= lon <= extent[1] - margin) and
                (extent[2] + margin <= lat <= extent[3] - margin)):
            continue
        # Check if this low is too close to any already plotted low
        too_close = any(np.hypot(y - py, x - px) < min_low_distance for py, px in plotted_low_points)
        if too_close:
            continue
        plotted_low_points.append((y, x))
        ax.text(
            lon, lat, "L",
            color='red', fontsize=16, fontweight='bold',
            ha='center', va='center', transform=ccrs.PlateCarree(),
            zorder=3, path_effects=[path_effects.Stroke(linewidth=1, foreground='white'), path_effects.Normal()]
        )
        ax.text(
            lon, lat - 0.7, f"{data_masked[y, x]:.0f}",
            color='red', fontsize=5, fontweight='bold',
            ha='center', va='top', transform=ccrs.PlateCarree(),
            zorder=3, path_effects=[path_effects.Stroke(linewidth=0.5, foreground='white'), path_effects.Normal()]
        )
        lows_plotted += 1

    # --- Highs detection ---
    # Only search for extrema within the plotted region
    mask = (
        (Lon2d >= extent[0]) & (Lon2d <= extent[1]) &
        (Lat2d >= extent[2]) & (Lat2d <= extent[3])
    )
    data_masked = np.where(mask, data2d, np.nan)

    # Find local maxima (Highs) >= 1020 hPa
    max_filt = ndimage.maximum_filter(data_masked, size=25, mode='constant', cval=np.nan)
    highs = (data_masked == max_filt) & ~np.isnan(data_masked) & (data_masked >= 1020)
    high_y, high_x = np.where(highs)
    highs_plotted = 0
    max_highs = 4
    min_high_distance = 60  # minimum grid points between plotted highs
    plotted_high_points = []
    for y, x in zip(high_y, high_x):
        if highs_plotted >= max_highs:
            break
        lon, lat = Lon2d[y, x], Lat2d[y, x]
        if not in_extent(lon, lat):
            continue
        # Check if this high is too close to any already plotted high
        too_close = False
        for py, px in plotted_high_points:
            if np.hypot(y - py, x - px) < min_high_distance:
                too_close = True
                break
        if too_close:
            continue
        plotted_high_points.append((y, x))
        ax.text(
            lon, lat, "H",
            color='blue', fontsize=16, fontweight='bold',
            ha='center', va='center', transform=ccrs.PlateCarree(),
            zorder=3, path_effects=[path_effects.Stroke(linewidth=1, foreground='white'), path_effects.Normal()]
        )
        ax.text(
            lon, lat-0.7, f"{data2d[y, x]:.0f}",
            color='blue', fontsize=5, fontweight='bold',
            ha='center', va='top', transform=ccrs.PlateCarree(),
            zorder=3, path_effects=[path_effects.Stroke(linewidth=0.5, foreground='white'), path_effects.Normal()]
        )
        highs_plotted += 1

    # Find local minima (Lows) <= 1006 hPa (lowered threshold for hurricanes)
    min_filt = ndimage.minimum_filter(data_masked, size=25, mode='constant', cval=np.nan)
    lows = (data_masked == min_filt) & ~np.isnan(data_masked) & (data_masked <= 1006)
    low_y, low_x = np.where(lows)
    lows_plotted = 0
    max_lows = 4
    min_low_distance = 60  # increased minimum grid points between plotted lows
    plotted_low_points = []

    for y, x in zip(low_y, low_x):
        if lows_plotted >= max_lows:
            break
        if (y, x) in plotted_high_points:
            continue  # Skip if already plotted as a High
        lon, lat = Lon2d[y, x], Lat2d[y, x]
        if not in_extent(lon, lat):
            continue
        # Check if this low is too close to any already plotted low
        too_close = False
        for py, px in plotted_low_points:
            if np.hypot(y - py, x - px) < min_low_distance:
                too_close = True
                break
        if too_close:
            continue

        ax.text(
            lon, lat, "L",
            color='red', fontsize=16, fontweight='bold',
            ha='center', va='center', transform=ccrs.PlateCarree(),
            zorder=3, path_effects=[path_effects.Stroke(linewidth=1, foreground='white'), path_effects.Normal()]
        )
        ax.text(
            lon, lat-0.7, f"{data2d[y, x]:.0f}",
            color='red', fontsize=5, fontweight='bold',
            ha='center', va='top', transform=ccrs.PlateCarree(),
            zorder=3, path_effects=[path_effects.Stroke(linewidth=0.5, foreground='white'), path_effects.Normal()]
        )
        plotted_low_points.append((y, x))
        lows_plotted += 1

    # Always plot the absolute minimum pressure as "L" if not already plotted and not too close to existing lows
    flat_data = data2d.flatten()
    min_idx = np.nanargmin(flat_data)
    min_y_full, min_x_full = np.unravel_index(min_idx, data2d.shape)

    if in_extent(Lon2d[min_y_full, min_x_full], Lat2d[min_y_full, min_x_full]):
        window = 2
        y0, x0 = min_y_full, min_x_full
        y1, y2 = max(0, y0 - window), min(data2d.shape[0], y0 + window + 1)
        x1, x2 = max(0, x0 - window), min(data2d.shape[1], x0 + window + 1)
        local_patch = data2d[y1:y2, x1:x2]
        grad_y, grad_x = np.gradient(local_patch)
        grad_mag = np.sqrt(grad_y**2 + grad_x**2)
        max_grad = np.nanmax(grad_mag)
        close_contour_threshold = 2.0  # hPa/grid, tune as needed

        already_plotted = any(np.hypot(min_y_full - py, min_x_full - px) < min_low_distance for py, px in plotted_low_points)
        if (not already_plotted) or (max_grad > close_contour_threshold and not already_plotted):
            if not already_plotted:
                ax.text(
                    Lon2d[min_y_full, min_x_full], Lat2d[min_y_full, min_x_full], "L",
                    color='red', fontsize=16, fontweight='bold',
                    ha='center', va='center', transform=ccrs.PlateCarree(),
                    zorder=3, path_effects=[path_effects.Stroke(linewidth=1, foreground='white'), path_effects.Normal()]
                )
                ax.text(
                    Lon2d[min_y_full, min_x_full], Lat2d[min_y_full, min_x_full]-0.7, f"{data2d[min_y_full, min_x_full]:.0f}",
                    color='red', fontsize=5, fontweight='bold',
                    ha='center', va='top', transform=ccrs.PlateCarree(),
                    zorder=3, path_effects=[path_effects.Stroke(linewidth=0.5, foreground='white'), path_effects.Normal()]
                )
                plotted_low_points.append((min_y_full, min_x_full))

    # Always plot an L at the location of the tightest pressure gradient (if extreme and not too close to existing lows)
    grad_y_full, grad_x_full = np.gradient(data2d)
    grad_mag_full = np.sqrt(grad_y_full**2 + grad_x_full**2)
    max_grad_idx = np.nanargmax(grad_mag_full)
    grad_y, grad_x = np.unravel_index(max_grad_idx, grad_mag_full.shape)
    max_grad_value = grad_mag_full[grad_y, grad_x]
    tight_gradient_threshold = 3.0  # hPa/grid, adjust as needed

    window = 2
    y1, y2 = max(0, grad_y - window), min(data2d.shape[0], grad_y + window + 1)
    x1, x2 = max(0, grad_x - window), min(data2d.shape[1], grad_x + window + 1)
    local_patch = data2d[y1:y2, x1:x2]
    if np.any(~np.isnan(local_patch)):
        local_min_idx = np.nanargmin(local_patch)
        local_min_y, local_min_x = np.unravel_index(local_min_idx, local_patch.shape)
        abs_min_y = y1 + local_min_y
        abs_min_x = x1 + local_min_x

        if (extent[0] <= Lon2d[abs_min_y, abs_min_x] <= extent[1] and
            extent[2] <= Lat2d[abs_min_y, abs_min_x] <= extent[3]):
            already_plotted = any(np.hypot(abs_min_y - py, abs_min_x - px) < min_low_distance for py, px in plotted_low_points)
            if (max_grad_value > tight_gradient_threshold) and not already_plotted:
                ax.text(
                    Lon2d[abs_min_y, abs_min_x], Lat2d[abs_min_y, abs_min_x], "L",
                    color='red', fontsize=16, fontweight='bold',
                    ha='center', va='center', transform=ccrs.PlateCarree(),
                    zorder=3, path_effects=[path_effects.Stroke(linewidth=1, foreground='white'), path_effects.Normal()]
                )
                ax.text(
                    Lon2d[abs_min_y, abs_min_x], Lat2d[abs_min_y, abs_min_x]-0.7, f"{data2d[abs_min_y, abs_min_x]:.0f}",
                    color='red', fontsize=5, fontweight='bold',
                    ha='center', va='top', transform=ccrs.PlateCarree(),
                    zorder=3, path_effects=[path_effects.Stroke(linewidth=0.5, foreground='white'), path_effects.Normal()]
                )
                plotted_low_points.append((abs_min_y, abs_min_x))

    # Plot county boundaries if available
    counties = load_county_boundaries(COUNTY_JSON_PATH)
    if counties is not None:
        counties = counties.to_crs(ccrs.PlateCarree().proj4_init)  # Ensure CRS matches the map
        counties.plot(ax=ax, edgecolor="gray", facecolor="none", linewidth=0.3, zorder=4)

    # Draw a black rectangle around the plot area (axes border)
    from matplotlib.patches import Rectangle
    # Get current axis limits
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    # Add rectangle patch as border
    rect = Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=2, edgecolor='black', facecolor='none', zorder=10, transform=ax.transData)
    ax.add_patch(rect)

    ax.set_axis_off()
    png_path = os.path.join(png_dir, f"usa_gfs_{step:03d}.png")
    plt.savefig(
        png_path,
        bbox_inches="tight",
        pad_inches=0.3,  # Reduced padding at the top and bottom
        dpi=600
    )
    plt.close(fig)
    print(f"Generated combined PNG: {png_path}")
    return png_path

# Main process
for step in forecast_steps:
    mslp_grib = get_mslp_grib(step)
    prate_grib = get_prate_grib(step)
    csnow_grib = get_csnow_grib(step)
    cfrzr_grib = get_cfrzr_grib(step)
    cicep_grib = get_cicep_grib(step)
    if mslp_grib and prate_grib:
        plot_combined(mslp_grib, prate_grib, step, csnow_grib, cfrzr_grib, cicep_grib)
        gc.collect()
        time.sleep(1)

# Delete all GRIB files in the updated grib_dir after PNGs are made
for f in os.listdir(grib_dir):
    file_path = os.path.join(grib_dir, f)
    if os.path.isfile(file_path):
        os.remove(file_path)

# --- Optimize all PNGs in the output directories ---
from PIL import Image

def optimize_png(filepath):
    try:
        with Image.open(filepath) as img:
            img = img.convert('P', palette=Image.ADAPTIVE, colors=256)
            img.save(filepath, optimize=True)
            print(f"Optimized PNG: {filepath}")
    except Exception as e:
        print(f"Failed to optimize {filepath}: {e}")

# Optimize PNGs
for f in os.listdir(png_dir):
    if f.lower().endswith('.png'):
        optimize_png(os.path.join(png_dir, f))

print("All PNGs optimized.")
