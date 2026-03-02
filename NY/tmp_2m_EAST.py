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


# --- Utility to fetch expanded region geojson and compute extent/boundary ---
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
    region_fips = ["23", "33", "50", "25", "44", "09", "36", "34", "42", "39", "51", "24", "10", "11", "54"]
    counties = gdf[gdf["fips"].str[:2].isin(region_fips)]
    if counties.empty:
        raise RuntimeError("No region counties found in GeoJSON.")
    minx, miny, maxx, maxy = counties.total_bounds
    pad_x = (maxx - minx) * padding_frac
    pad_y = (maxy - miny) * padding_frac
    extent = [minx - pad_x, maxx + pad_x, miny - pad_y, maxy + pad_y]

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

# Output directories (use new region folder)
grib_dir = os.path.join(BASE_DIR, "tmp_2m_EAST", "grib")
png_dir = os.path.join(BASE_DIR, "tmp_2m_EAST", "png")
os.makedirs(grib_dir, exist_ok=True)
os.makedirs(png_dir, exist_ok=True)

# NEW: directory to store previous-run averaged arrays for accuracy comparisons
prev_dir = os.path.join(BASE_DIR, "tmp_2m_EAST", "prev")
os.makedirs(prev_dir, exist_ok=True)

# File to track completed forecast steps for the current run
processed_steps_file = os.path.join(BASE_DIR, "tmp_2m_EAST", "processed_steps.txt")

# Clear the processed steps file at the start of a new run
if os.path.exists(processed_steps_file):
    os.remove(processed_steps_file)

# HRRR URL and variables
base_url_hrrr = "https://nomads.ncep.noaa.gov/cgi-bin/filter_hrrr_2d.pl"
variable_tmp = "TMP"  # 2-meter temperature

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
# Levels and colormaps for temperature in Fahrenheit
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

# Adjust forecast steps to process in chunks of 3
forecast_steps = list(range(1, 49, 48))  # f01, f04, f07, ..., f46

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

# Modify the get_hrrr_grib function to handle multiple steps
def get_hrrr_grib(steps, variable):
    global date_str, hour_str  # Allow fallback to modify these variables
    file_paths = []
    for step in steps:
        file_name = f"hrrr.t{hour_str}z.wrfsfcf{step:02d}.grib2"
        file_path = os.path.join(grib_dir, f"{variable.lower()}_{file_name}")
        url = (
            f"{base_url_hrrr}?file={file_name}"
            f"&lev_2_m_above_ground=on"
            f"&var_{variable}=on"
            f"&dir=%2Fhrrr.{date_str}%2Fconus"
        )
        if download_grib(url, file_path):
            file_paths.append(file_path)
        else:
            # Fallback to the previous run if the most recent run fails
            print(f"Falling back to previous run for step {step}, variable {variable}")
            fallback_date_str, fallback_hour_str = get_run_strings(previous_run_time)
            fallback_file_name = f"hrrr.t{fallback_hour_str}z.wrfsfcf{step:02d}.grib2"
            fallback_file_path = os.path.join(grib_dir, f"{variable.lower()}_{fallback_file_name}")
            fallback_url = (
                f"{base_url_hrrr}?file={fallback_file_name}"
                f"&lev_2_m_above_ground=on"
                f"&var_{variable}=on"
                f"&dir=%2Fhrrr.{fallback_date_str}%2Fconus"
            )
            if download_grib(fallback_url, fallback_file_path):
                # Update global date_str and hour_str to fallback values
                date_str, hour_str = fallback_date_str, fallback_hour_str
                file_paths.append(fallback_file_path)
            else:
                print(f"Failed to download {variable} for both runs (step {step})")
    return file_paths

# --- Plotting function ---
def plot_tmp_2m(tmp_path, step):
    try:
        # Use dask chunking for lazy loading
        ds_tmp = xr.open_dataset(tmp_path, engine="cfgrib", chunks={})
    except Exception as e:
        print(f"Error opening dataset: {e}")
        return None

    # Extract arrays
    tmp = ds_tmp.get('t2m')
    if tmp is None:
        print("Required variable not in dataset")
        ds_tmp.close()
        return None
    tmp = (tmp.values - 273.15) * 9/5 + 32  # Convert from Kelvin to Fahrenheit
    ds_tmp.close()

    lats = ds_tmp['latitude'].values
    lons = ds_tmp['longitude'].values
    lons_plot = np.where(lons > 180, lons - 360, lons)

    if lats.ndim == 1 and lons.ndim == 1:
        Lon2d, Lat2d = np.meshgrid(lons_plot, lats)
        tmp2d = tmp.squeeze()
    else:
        Lon2d, Lat2d = lons_plot, lats
        tmp2d = tmp.squeeze()

    # Do not mask weather data to region; plot full grid

    # Title/time calculation — use timezone-aware conversion so DST is handled
    base_time = datetime.strptime(f"{date_str} {hour_str}", "%Y%m%d %H")
    # treat base_time as UTC
    base_time_utc = base_time.replace(tzinfo=timezone.utc)
    valid_time = base_time_utc + timedelta(hours=step)  # Forecast valid time in UTC
    # convert valid_time to America/New_York to get local hour and weekday (handles DST)
    local_valid = valid_time.astimezone(ZoneInfo('America/New_York'))
    local_time = local_valid.strftime('%I %p')
    day_of_week = local_valid.strftime('%A')


    title = (
        f"HRRR 2m Temperature — Northeast/Mid-Atlantic US\n"
        f"Valid: {valid_time.strftime('%Y-%m-%d %HZ')} ({day_of_week}, {local_time})  "
        f"Run: {hour_str}Z  Forecast Hour: {step}"
    )

    # Plotting setup
    fig = plt.figure(figsize=(13, 11), dpi=300, facecolor='white')
    fig.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.16)
    ax = plt.axes(projection=ccrs.PlateCarree(), facecolor='white')
    ax.set_extent(REGION_EXTENT, crs=ccrs.PlateCarree())
    ax.set_title(title, fontsize=11, fontweight='bold', pad=10)

    # Base map
    ax.add_feature(cfeature.LAND, facecolor='white')
    ax.add_feature(cfeature.OCEAN, facecolor='white')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.7, edgecolor='black')
    ax.add_feature(cfeature.LAKES, facecolor='lightblue', edgecolor='black')

    # TMP plotting
    mesh = ax.contourf(
        Lon2d, Lat2d, tmp2d,
        levels=temp_levels,
        cmap=custom_cmap,
        norm=temp_norm,
        extend='both',
        transform=ccrs.PlateCarree(),
        alpha=0.8,
        zorder=1
    )

    # Optionally, add temperature values as text (commented out for clarity)
    # for lat in np.arange(np.floor(Lat2d.min()), np.ceil(Lat2d.max()), 0.5):
    #     for lon in np.arange(np.floor(Lon2d.min()), np.ceil(Lon2d.max()), 0.5):
    #         lat_idx = (np.abs(Lat2d[:, 0] - lat)).argmin()
    #         lon_idx = (np.abs(Lon2d[0, :] - lon)).argmin()
    #         if not np.isnan(tmp2d[lat_idx, lon_idx]):
    #             ax.text(
    #                 Lon2d[lat_idx, lon_idx], Lat2d[lat_idx, lon_idx], f"{tmp2d[lat_idx, lon_idx]:.1f}",
    #                 fontsize=5, color="black", ha="center", va="center",
    #                 transform=ccrs.PlateCarree(), zorder=10
    #             )

    # Colorbar
    cbar_ax = fig.add_axes([0.05, 0.08, 0.9, 0.02])  # Adjusted to match plot width
    cbar = plt.colorbar(mesh, cax=cbar_ax, orientation='horizontal')
    cbar.set_label("2m Temperature (°F)", fontsize=8)
    cbar.set_ticks(temp_levels)  # Ensure every tick is shown
    cbar.ax.tick_params(labelsize=6)


    # Overlay region counties and state outlines
    try:
        region_gdf.plot(ax=ax, facecolor="none", edgecolor="gray", linewidth=0.3, zorder=7)
        region_states_gdf.boundary.plot(ax=ax, edgecolor="#000000", linewidth=1.0, zorder=8)
    except Exception as e:
        print(f"Error plotting overlays: {e}")

    # Add attribution text at the very bottom right of the map area, stacked tightly

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

    # Save PNG
    png_path = os.path.join(png_dir, f"hrrr_tmp_2m_EAST_{step:02d}.png")
    plt.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)

    # Explicitly delete large objects and collect garbage
    del tmp, Lon2d, Lat2d, tmp2d, mesh, fig, ax
    gc.collect()

    print(f"Generated PNG: {png_path}")
    return png_path

# Clear the folders at the start of the script
def clear_folder(folder_path):
    if os.path.exists(folder_path):
        for f in os.listdir(folder_path):
            file_path = os.path.join(folder_path, f)
            if os.path.isfile(file_path):
                os.remove(file_path)
        print(f"Cleared folder: {folder_path}")

clear_folder(grib_dir)
clear_folder(png_dir)

# Main process
for step in forecast_steps:
    steps_to_process = range(step, step + 48)  # Process 3 steps at a time
    tmp_gribs = get_hrrr_grib(steps_to_process, "TMP")
    for i, tmp_grib in enumerate(tmp_gribs):
        if tmp_grib:
            plot_tmp_2m(tmp_grib, step + i)

            # Explicitly delete processed GRIB file and collect garbage
            del tmp_grib
            gc.collect()

print("HRRR 2m Temperature processing complete.")
