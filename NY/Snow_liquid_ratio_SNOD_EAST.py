# --- Utility to fetch Northeast/Mid-Atlantic/Ohio/VA geojson and compute extent/boundary ---
import os
import requests
import shutil
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm, ListedColormap
from matplotlib import patheffects
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import gc  # Add garbage collection module

# --- Utility to fetch Northeast/Mid-Atlantic/Ohio/VA geojson and compute extent/boundary ---
def get_eastern_geodata(padding_frac=0.09):
    state_fips = [
        '23', '33', '50', '25', '44', '09', '36', '34', '42', '39', '51', '24', '10', '11', '54'
    ]
    url_counties = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
    r = requests.get(url_counties)
    r.raise_for_status()
    geojson = r.json()
    for feat in geojson.get("features", []):
        feat.setdefault("properties", {})
        feat["properties"]["fips"] = feat.get("id", "")
    gdf = gpd.GeoDataFrame.from_features(geojson["features"])
    gdf = gdf.set_crs("EPSG:4326")
    gdf["fips"] = gdf["fips"].astype(str)
    region_gdf = gdf[gdf["fips"].str[:2].isin(state_fips)]
    if region_gdf.empty:
        raise RuntimeError("No region counties found in GeoJSON.")
    minx, miny, maxx, maxy = region_gdf.total_bounds
    pad_x = (maxx - minx) * padding_frac
    pad_y = (maxy - miny) * padding_frac
    extent = [minx - pad_x, maxx + pad_x, miny - pad_y, maxy + pad_y]
    url_states = "https://www2.census.gov/geo/tiger/GENZ2022/shp/cb_2022_us_state_20m.zip"
    states_gdf = gpd.read_file(url_states)
    region_states_gdf = states_gdf[states_gdf['STATEFP'].isin(state_fips)]
    region_outline = region_states_gdf.unary_union
    return region_gdf, extent, region_outline, region_states_gdf

region_gdf, REGION_EXTENT, region_outline, region_states_gdf = get_eastern_geodata()

BASE_DIR = '/var/data'
grib_dir = os.path.join(BASE_DIR, "SNOD_EAST", "grib")
png_dir = os.path.join(BASE_DIR, "SNOD_EAST", "png")
os.makedirs(grib_dir, exist_ok=True)
os.makedirs(png_dir, exist_ok=True)
processed_steps_file = os.path.join(BASE_DIR, "SNOD_EAST", "processed_steps.txt")
if os.path.exists(processed_steps_file):
    os.remove(processed_steps_file)

base_url_hrrr = "https://nomads.ncep.noaa.gov/cgi-bin/filter_hrrr_2d.pl"
variable_snod = "sde"  # Snow Depth

# Adjust available hours to include 03Z, 09Z, 15Z, 21Z with a max forecast range of 18 hours
available_hours = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
max_forecast_hours = {1, 2, 3, 4, 5,7,8, 9, 10, 11, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23}  # These runs only go out to 18 hours

# Function to validate if a run time is accessible
def is_valid_run(run_time):
    """Check if HRRR data for the given run time is accessible."""
    test_date_str = run_time.strftime("%Y%m%d")
    test_hour_str = run_time.strftime("%H")
    test_file_name = f"hrrr.t{test_hour_str}z.wrfsfcf01.grib2"
    test_url = (
        f"{base_url_hrrr}?file={test_file_name}"
        f"&lev_surface=on&lev_mean_sea_level=on"
        f"&var_MSLMA=on"
        f"&dir=%2Fhrrr.{test_date_str}%2Fconus"
    )
    try:
        response = requests.head(test_url, timeout=5)
        return response.status_code == 200
    except Exception:
        return False

# Function to get the forecast steps based on the run hour
def get_forecast_steps(run_hour):
    if run_hour in {3, 9, 13, 14, 15, 16, 21}:  # Limit to 18-hour forecast for 03Z, 09Z, 13Z, 14Z, 15Z, 16Z, 21Z
        return list(range(1, 19))  # 18-hour forecast
    return list(range(1, 49, 1))  # Default 48-hour forecast in 3-hour steps

# Calculate the most recent HRRR run dynamically
current_utc_time = datetime.utcnow()
most_recent_run_time = None

# Iterate backward to find the most recent valid run, checking all available hours
for offset in range(24):  # Check up to 24 hours back
    candidate_hour = (current_utc_time.hour - offset) % 24
    if candidate_hour in available_hours:
        candidate_time = current_utc_time.replace(hour=candidate_hour, minute=0, second=0, microsecond=0)
        # Check if the run is valid (e.g., file exists or accessible)
        if is_valid_run(candidate_time):  # Replace with actual validation logic
            most_recent_run_time = candidate_time
            break

# If no valid run found, fall back to previous run (6 hours earlier)
if most_recent_run_time is None:
    print("No valid run time found in the available hours. Searching for fallback run hour one hour at a time.")
    fallback_found = False
    for offset in range(1, 25):  # Search up to 24 hours back, one hour at a time
        candidate_time = current_utc_time - timedelta(hours=offset)
        candidate_hour = candidate_time.hour
        if candidate_hour in available_hours:
            candidate_time = candidate_time.replace(minute=0, second=0, microsecond=0)
            if is_valid_run(candidate_time):
                most_recent_run_time = candidate_time
                print(f"Using fallback run time: {most_recent_run_time}")
                fallback_found = True
                break
    if not fallback_found:
        raise ValueError("No valid run time found, including fallback.")

forecast_steps = get_forecast_steps(most_recent_run_time.hour)

# Define previous_run_time as the most recent run time minus the forecast interval
previous_run_time = most_recent_run_time - timedelta(hours=6)

# Function to get date and hour strings for a given run time
def get_run_strings(run_time):
    return run_time.strftime("%Y%m%d"), run_time.strftime("%H")

# Start with the most recent run
date_str, hour_str = get_run_strings(most_recent_run_time)


# Snowfall colormap and levels (inches, 8:1 ratio)  

snow_breaks = [
    0, 0.1, 1, 2, 3, 4, 5, 6, 7, 8,
    10, 12, 16, 20, 24, 36, 48, 56,
    64, 72, 84, 100
]

snow_colors = [
    "#ffffff", "#0d1a4a", "#1565c0", "#42a5f5", "#90caf9", "#e3f2fd",
    "#b39ddb", "#7e57c2", "#512da8", "#c2185b", "#f06292", "#81c784",
    "#388e3c", "#1b5e20", "#bdbdbd", "#757575", "#424242", "#212121",
    "#F4F805", "#FDAE04", "#F70909"
]
snow_cmap = ListedColormap(snow_colors)
snow_norm = BoundaryNorm(snow_breaks, len(snow_colors))

forecast_steps = get_forecast_steps(most_recent_run_time.hour)

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

def get_hrrr_grib(steps, variable):
    global date_str, hour_str
    file_paths = []
    for step in steps:
        file_name = f"hrrr.t{hour_str}z.wrfsfcf{step:02d}.grib2"
        file_path = os.path.join(grib_dir, f"{variable.lower()}_{file_name}")
        url = (
            f"{base_url_hrrr}?file={file_name}"
            f"&lev_surface=on"
            f"&var_{variable}=on"
            f"&dir=%2Fhrrr.{date_str}%2Fconus"
        )
        if download_grib(url, file_path):
            file_paths.append(file_path)
        else:
            print(f"Falling back to previous run for step {step}, variable {variable}")
            fallback_date_str, fallback_hour_str = get_run_strings(previous_run_time)
            fallback_file_name = f"hrrr.t{fallback_hour_str}z.wrfsfcf{step:02d}.grib2"
            fallback_file_path = os.path.join(grib_dir, f"{variable.lower()}_{fallback_file_name}")
            fallback_url = (
                f"{base_url_hrrr}?file={fallback_file_name}"
                f"&lev_surface=on"
                f"&var_{variable}=on"
                f"&dir=%2Fhrrr.{fallback_date_str}%2Fconus"
            )
            if download_grib(fallback_url, fallback_file_path):
                date_str, hour_str = fallback_date_str, fallback_hour_str
                file_paths.append(fallback_file_path)
            else:
                print(f"Failed to download {variable} for both runs (step {step})")
    return file_paths

# --- Plotting function ---
def plot_snod_surface(snod_path, step):
    try:
        ds_snod = xr.open_dataset(
            snod_path,
            engine="cfgrib",
            filter_by_keys={"stepType": "instant"},
            chunks={}
        )
    except Exception as e:
        print(f"Error opening dataset: {e}")
        return None
    if 'sde' not in ds_snod.variables:
        print(f"Required variable 'sde' not found in dataset. Available variables: {list(ds_snod.variables.keys())}")
        return None
    # Convert from meters to inches
    snod = ds_snod['sde'].values * 39.3701
    ds_snod.close()
    lats = ds_snod['latitude'].values
    lons = ds_snod['longitude'].values
    lons_plot = np.where(lons > 180, lons - 360, lons)
    if lats.ndim == 1 and lons.ndim == 1:
        Lon2d, Lat2d = np.meshgrid(lons_plot, lats)
        snod2d = snod.squeeze()
    else:
        Lon2d, Lat2d = lons_plot, lats
        snod2d = snod.squeeze()
    base_time = datetime.strptime(f"{date_str} {hour_str}", "%Y%m%d %H")
    base_time_utc = base_time.replace(tzinfo=timezone.utc)
    valid_time = base_time_utc + timedelta(hours=step)
    local_valid = valid_time.astimezone(ZoneInfo('America/New_York'))
    local_time = local_valid.strftime('%I %p')
    day_of_week = local_valid.strftime('%A')
    title = (
        f"HRRR Surface Snow Depth (SNOD) — Northeast US\n"
        f"Valid: {valid_time.strftime('%Y-%m-%d %HZ')} ({day_of_week}, {local_time})  "
        f"Run: {hour_str}Z  Forecast Hour: {step}"
    )
    fig = plt.figure(figsize=(12, 9), dpi=300, facecolor='white')
    fig.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.16)
    ax = plt.axes(projection=ccrs.PlateCarree(), facecolor='white')
    ax.set_extent(REGION_EXTENT, crs=ccrs.PlateCarree())
    ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
    ax.add_feature(cfeature.LAND, facecolor='white')
    ax.add_feature(cfeature.OCEAN, facecolor='white')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.7, edgecolor='black')
    ax.add_feature(cfeature.LAKES, facecolor='lightblue', edgecolor='black')
    mesh = ax.contourf(
        Lon2d, Lat2d, snod2d,
        levels=snow_breaks,
        cmap=snow_cmap,
        norm=snow_norm,
        extend='both',
        transform=ccrs.PlateCarree(),
        alpha=0.8,
        zorder=1
    )
    cbar_ax = fig.add_axes([0.05, 0.08, 0.9, 0.02])
    cbar = plt.colorbar(mesh, cax=cbar_ax, orientation='horizontal')
    cbar.set_label("Snow Depth (inches)", fontsize=8)
    cbar.set_ticks(snow_breaks)
    cbar.ax.tick_params(labelsize=6)
    try:
        region_gdf.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=0.3, zorder=7)
        region_states_gdf.boundary.plot(ax=ax, edgecolor="#000000", linewidth=1.0, zorder=8)
    except Exception as e:
        print(f"Error plotting overlays: {e}")
    margin_x = (REGION_EXTENT[1] - REGION_EXTENT[0]) * 0.01
    margin_y = (REGION_EXTENT[3] - REGION_EXTENT[2]) * 0.01
    text_x = REGION_EXTENT[1] - margin_x
    text_y_base = REGION_EXTENT[2] + margin_y
    line_spacing = (REGION_EXTENT[3] - REGION_EXTENT[2]) * 0.025
    ax.text(
        text_x, text_y_base + line_spacing, "Images by Jack Fordyce",
        fontsize=7, color="black", ha="right", va="bottom",
        fontweight="normal", alpha=0.85,
        transform=ccrs.PlateCarree(),
        zorder=20,
        path_effects=[
            patheffects.Stroke(linewidth=1, foreground='white'),
            patheffects.Normal()
        ]
    )
    ax.text(
        text_x, text_y_base, "NYWeatherModels.com",
        fontsize=7, color="black", ha="right", va="bottom",
        fontweight="normal", alpha=0.85,
        transform=ccrs.PlateCarree(),
        zorder=20,
        path_effects=[
            patheffects.Stroke(linewidth=1, foreground='white'),
            patheffects.Normal()
        ]
    )
    png_path = os.path.join(png_dir, f"hrrr_SNOD_EAST_{step:02d}.png")
    plt.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    del snod, Lon2d, Lat2d, snod2d, mesh, fig, ax
    gc.collect()
    print(f"Generated PNG: {png_path}")
    return png_path

def clear_folder(folder_path):
    if os.path.exists(folder_path):
        for f in os.listdir(folder_path):
            file_path = os.path.join(folder_path, f)
            if os.path.isfile(file_path):
                os.remove(file_path)
        print(f"Cleared folder: {folder_path}")

clear_folder(grib_dir)
clear_folder(png_dir)

for step in forecast_steps:
    if step > max(forecast_steps):
        break
    snod_gribs = get_hrrr_grib([step], "SNOD")
    for i, snod_grib in enumerate(snod_gribs):
        if snod_grib:
            plot_snod_surface(snod_grib, step + i)
            del snod_grib
            gc.collect()
print("HRRR SNOD Surface Snow Depth processing complete.")
