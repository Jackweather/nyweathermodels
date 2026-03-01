import os
import requests
import shutil
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
from matplotlib import patheffects
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import gc  # Add garbage collection module

# --- Utility to fetch NY geojson and compute extent/boundary ---

# --- Utility to fetch Northeast US geojson and compute extent/boundary ---

# --- Utility to fetch Northeast+MidAtlantic geojson and compute extent/boundary ---
def get_eastern_geodata(padding_frac=0.05):
    url = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
    r = requests.get(url)
    r.raise_for_status()
    geojson = r.json()

    for feat in geojson.get("features", []):
        feat.setdefault("properties", {})
        feat["properties"]["fips"] = feat.get("id", "")

    gdf = gpd.GeoDataFrame.from_features(geojson["features"])
    gdf = gdf.set_crs("EPSG:4326")
    gdf["fips"] = gdf["fips"].astype(str)

    # FIPS state codes for Northeast + Mid-Atlantic + OH/VA
    # ME(23), NH(33), VT(50), MA(25), RI(44), CT(09), NY(36), NJ(34), PA(42), OH(39), VA(51), MD(24), DE(10), DC(11), WV(54)
    region_fips = ["23", "33", "50", "25", "44", "09", "36", "34", "42", "39", "51", "24", "10", "11", "54"]
    counties = gdf[gdf["fips"].str[:2].isin(region_fips)]
    if counties.empty:
        raise RuntimeError("No region counties found in GeoJSON.")
    minx, miny, maxx, maxy = counties.total_bounds
    pad_x = (maxx - minx) * padding_frac
    pad_y = (maxy - miny) * padding_frac
    extent = [minx - pad_x, maxx + pad_x, miny - pad_y, maxy + pad_y]

    # Get state outlines for the region
    state_names = [
        "Maine", "New Hampshire", "Vermont", "Massachusetts", "Rhode Island", "Connecticut",
        "New York", "New Jersey", "Pennsylvania", "Ohio", "Virginia", "Maryland", "Delaware", "District of Columbia", "West Virginia"
    ]
    census_states_url = "https://eric.clst.org/assets/wiki/uploads/Stuff/gz_2010_us_040_00_500k.json"
    states_census_gdf = gpd.read_file(census_states_url)
    states_census_gdf = states_census_gdf[states_census_gdf["NAME"].isin(state_names)]
    state_outline = states_census_gdf.unary_union
    return counties, extent, state_outline, states_census_gdf

# Acquire region geodata once (cache shapes in memory)
region_gdf, REGION_EXTENT, region_outline, region_states_gdf = get_eastern_geodata()

# Set base directory for HRRR output
BASE_DIR = '/var/data'


grib_dir = os.path.join(BASE_DIR, "vis_EAST", "grib")
png_dir = os.path.join(BASE_DIR, "vis_EAST", "png")
os.makedirs(grib_dir, exist_ok=True)
os.makedirs(png_dir, exist_ok=True)

prev_dir = os.path.join(BASE_DIR, "vis_EAST", "prev")
os.makedirs(prev_dir, exist_ok=True)

processed_steps_file = os.path.join(BASE_DIR, "vis_EAST", "processed_steps.txt")
if os.path.exists(processed_steps_file):
    os.remove(processed_steps_file)

base_url_hrrr = "https://nomads.ncep.noaa.gov/cgi-bin/filter_hrrr_2d.pl"
variable_vis = "VIS"  # Visibility

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

if most_recent_run_time is None:
    raise ValueError("No valid run time found in the available hours.")

forecast_steps = get_forecast_steps(most_recent_run_time.hour)

# Define previous_run_time as the most recent run time minus the forecast interval
previous_run_time = most_recent_run_time - timedelta(hours=6)

# Function to get date and hour strings for a given run time
def get_run_strings(run_time):
    return run_time.strftime("%Y%m%d"), run_time.strftime("%H")

# Start with the most recent run
date_str, hour_str = get_run_strings(most_recent_run_time)
# Snowfall colormap and levels (inches, 8:1 ratio)

# Meteorology-friendly visibility levels and colormap
# 0–0.25 mi (Dense fog): Dark purple
# 0.25–0.5 mi: Red
# 0.5–1 mi: Orange
# 1–3 mi: Yellow
# 3–6 mi: Light green
# 6–10 mi: Light blue
# 10+ mi: White
vis_levels_miles = [0, 0.25, 0.5, 1, 3, 6, 10]
custom_cmap = LinearSegmentedColormap.from_list(
    "vis_met_cmap",
    [
        "#3f007d",  # dark purple (0–0.25)
        "#e31a1c",  # red (0.25–0.5)
        "#ff7f00",  # orange (0.5–1)
        "#ffff33",  # yellow (1–3)
        "#b2df8a",  # light green (3–6)
        "#a6cee3",  # light blue (6–10)
        "#ffffff"   # white (10+)
    ],
    N=256
)
vis_norm = BoundaryNorm(vis_levels_miles + [1000], custom_cmap.N)  # 1000 is a dummy upper bound for 10+

forecast_steps = list(range(1, 49, 48))

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



def plot_vis(vis_path, step):
    try:
        ds_vis = xr.open_dataset(vis_path, engine="cfgrib", chunks={})
    except Exception as e:
        print(f"Error opening dataset: {e}")
        return None

    vis = ds_vis.get('vis')
    if vis is None:
        print("Required variable not in dataset")
        ds_vis.close()
        return None
    vis = vis.values  # meters
    ds_vis.close()
    # Convert meters to miles for plotting (0 = clear, higher = more obscured)
    vis = vis / 1609.34

    lats = ds_vis['latitude'].values
    lons = ds_vis['longitude'].values
    lons_plot = np.where(lons > 180, lons - 360, lons)

    if lats.ndim == 1 and lons.ndim == 1:
        Lon2d, Lat2d = np.meshgrid(lons_plot, lats)
        vis2d = vis.squeeze()
    else:
        Lon2d, Lat2d = lons_plot, lats
        vis2d = vis.squeeze()

    # Do not mask weather data to region; plot full grid

    base_time = datetime.strptime(f"{date_str} {hour_str}", "%Y%m%d %H")
    base_time_utc = base_time.replace(tzinfo=timezone.utc)
    valid_time = base_time_utc + timedelta(hours=step)
    local_valid = valid_time.astimezone(ZoneInfo('America/New_York'))
    local_time = local_valid.strftime('%I %p')
    day_of_week = local_valid.strftime('%A')

    title = (
        f"HRRR Surface Visibility — Northeast/Mid-Atlantic US\n"
        f"Valid: {valid_time.strftime('%Y-%m-%d %HZ')} ({day_of_week}, {local_time})  "
        f"Run: {hour_str}Z  Forecast Hour: {step}"
    )

    fig = plt.figure(figsize=(13, 11), dpi=300, facecolor='white')
    fig.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.16)
    ax = plt.axes(projection=ccrs.PlateCarree(), facecolor='white')
    ax.set_extent(REGION_EXTENT, crs=ccrs.PlateCarree())
    ax.set_title(title, fontsize=11, fontweight='bold', pad=10)

    ax.add_feature(cfeature.LAND, facecolor='white')
    ax.add_feature(cfeature.OCEAN, facecolor='white')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.7, edgecolor='black')
    ax.add_feature(cfeature.LAKES, facecolor='lightblue', edgecolor='black')

    mesh = ax.contourf(
        Lon2d, Lat2d, vis2d,
        levels=vis_levels_miles + [1000],  # 1000 is a dummy upper bound for 10+
        cmap=custom_cmap,
        norm=vis_norm,
        extend='max',
        transform=ccrs.PlateCarree(),
        alpha=0.8,
        zorder=1
    )

    # Plot all region counties and state outlines (cached)
    try:
        region_gdf.plot(ax=ax, facecolor="none", edgecolor="gray", linewidth=0.3, zorder=7)
        region_states_gdf.boundary.plot(ax=ax, edgecolor="#000000", linewidth=1.0, zorder=8)
    except Exception as e:
        print(f"Error plotting overlays: {e}")

    cbar_ax = fig.add_axes([0.05, 0.08, 0.9, 0.02])
    cbar = plt.colorbar(mesh, cax=cbar_ax, orientation='horizontal', extend='max')
    cbar.set_label("Surface Visibility (miles, 10+ = clear)", fontsize=8)
    cbar.set_ticks(vis_levels_miles)
    cbar.ax.tick_params(labelsize=6)

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
        text_x, text_y_base, "Truelocalwx.com",
        fontsize=7, color="black", ha="right", va="bottom",
        fontweight="normal", alpha=0.85,
        transform=ccrs.PlateCarree(),
        zorder=20,
        path_effects=[
            patheffects.Stroke(linewidth=1, foreground='white'),
            patheffects.Normal()
        ]
    )

    png_path = os.path.join(png_dir, f"hrrr_vis_EAST_{step:02d}.png")
    plt.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)

    del vis, Lon2d, Lat2d, vis2d, mesh, fig, ax
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
    steps_to_process = range(step, step + 48)
    vis_gribs = get_hrrr_grib(steps_to_process, "VIS")
    for i, vis_grib in enumerate(vis_gribs):
        if vis_grib:
            plot_vis(vis_grib, step + i)
            del vis_grib
            gc.collect()

print("HRRR Surface Visibility processing complete.")
