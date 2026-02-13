import os
import requests
import shutil
from datetime import datetime, timedelta
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import gc  # Add garbage collection module

# --- Utility to fetch NY geojson and compute extent/boundary ---
def get_ny_geodata(padding_frac=0.09):
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
    ny = gdf[gdf["fips"].str.startswith("36")]
    if ny.empty:
        raise RuntimeError("No New York counties found in GeoJSON.")
    minx, miny, maxx, maxy = ny.total_bounds
    pad_x = (maxx - minx) * padding_frac
    pad_y = (maxy - miny) * padding_frac
    extent = [minx - pad_x, maxx + pad_x, miny - pad_y, maxy + pad_y]
    state_outline = ny.unary_union
    return ny, extent, state_outline

# Acquire NY geodata once
ny_gdf, NY_EXTENT, ny_state_outline = get_ny_geodata()

# Set base directory for HRRR output
BASE_DIR = '/var/data'

# Output directories
grib_dir = os.path.join(BASE_DIR, "wind_10m_NY", "grib")
png_dir = os.path.join(BASE_DIR, "wind_10m_NY", "png")
os.makedirs(grib_dir, exist_ok=True)
os.makedirs(png_dir, exist_ok=True)

# Clear the grib and png directories at the start of the script
for folder in [grib_dir, png_dir]:
    for file_name in os.listdir(folder):
        file_path = os.path.join(folder, file_name)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)  # Remove the file
                print(f"Deleted: {file_path}")
        except Exception as e:
            print(f"Failed to delete {file_path}: {e}")

# HRRR URL and variables
base_url_hrrr = "https://nomads.ncep.noaa.gov/cgi-bin/filter_hrrr_2d.pl"
variables = ["UGRD", "VGRD"]  # U and V wind components

# Adjust available hours to include 03Z, 09Z, 15Z, 21Z with a max forecast range of 18 hours
available_hours = [0, 3, 6, 9, 12, 15, 18, 21]
max_forecast_hours = {3, 9, 15, 21}  # These runs only go out to 18 hours

# Function to get the forecast steps based on the run hour
def get_forecast_steps(run_hour):
    if run_hour in max_forecast_hours:  # Limit to 18-hour forecast for specific hours
        return list(range(1, 19))  # 18-hour forecast
    return list(range(1, 49, 1))  # Default 48-hour forecast in 1-hour steps

# Calculate the most recent HRRR run dynamically
current_utc_time = datetime.utcnow()
most_recent_run_hour = max(h for h in available_hours if h <= current_utc_time.hour)
most_recent_run_time = current_utc_time.replace(hour=most_recent_run_hour, minute=0, second=0, microsecond=0)
forecast_steps = get_forecast_steps(most_recent_run_hour)

# Define previous_run_time as the most recent run time minus the forecast interval
previous_run_time = most_recent_run_time - timedelta(hours=6)

# Function to get date and hour strings for a given run time
def get_run_strings(run_time):
    return run_time.strftime("%Y%m%d"), run_time.strftime("%H")

# Start with the most recent run
date_str, hour_str = get_run_strings(most_recent_run_time)

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

# Modify the get_hrrr_grib function to handle fallback logic
def get_hrrr_grib(steps, variables):
    global date_str, hour_str  # Allow fallback to modify these variables
    file_paths = []
    for step in steps:
        step_files = {}
        for variable in variables:
            file_name = f"hrrr.t{hour_str}z.wrfsfcf{step:02d}.grib2"
            file_path = os.path.join(grib_dir, f"{variable.lower()}_{file_name}")
            url = (
                f"{base_url_hrrr}?file={file_name}"
                f"&lev_10_m_above_ground=on"
                f"&var_{variable}=on"
                f"&dir=%2Fhrrr.{date_str}%2Fconus"
            )
            if download_grib(url, file_path):
                step_files[variable] = file_path
            else:
                # Fallback to the previous run if the most recent run fails
                print(f"Falling back to previous run for step {step}, variable {variable}")
                fallback_date_str, fallback_hour_str = get_run_strings(previous_run_time)
                fallback_file_name = f"hrrr.t{fallback_hour_str}z.wrfsfcf{step:02d}.grib2"
                fallback_file_path = os.path.join(grib_dir, f"{variable.lower()}_{fallback_file_name}")
                fallback_url = (
                    f"{base_url_hrrr}?file={fallback_file_name}"
                    f"&lev_10_m_above_ground=on"
                    f"&var_{variable}=on"
                    f"&dir=%2Fhrrr.{fallback_date_str}%2Fconus"
                )
                if download_grib(fallback_url, fallback_file_path):
                    # Update global date_str and hour_str to fallback values
                    date_str, hour_str = fallback_date_str, fallback_hour_str
                    step_files[variable] = fallback_file_path
                else:
                    print(f"Failed to download {variable} for both runs (step {step})")
        if len(step_files) == len(variables):
            file_paths.append(step_files)
    return file_paths

# --- Plotting function ---
def plot_wind_10m(u_path, v_path, step):
    try:
        ds_u = xr.open_dataset(u_path, engine="cfgrib")
        ds_v = xr.open_dataset(v_path, engine="cfgrib")
    except Exception as e:
        print(f"Error opening dataset: {e}")
        return None

    # Extract arrays
    u = ds_u.get('u10')
    v = ds_v.get('v10')
    if u is None or v is None:
        print("Required variables not in dataset")
        return None

    lats = ds_u['latitude'].values
    lons = ds_u['longitude'].values
    lons_plot = np.where(lons > 180, lons - 360, lons)

    if lats.ndim == 1 and lons.ndim == 1:
        Lon2d, Lat2d = np.meshgrid(lons_plot, lats)
        u2d, v2d = u.squeeze(), v.squeeze()
    else:
        Lon2d, Lat2d = lons_plot, lats
        u2d, v2d = u.squeeze(), v.squeeze()

    # Mask to New York polygon
    from matplotlib.path import Path
    polys = list(ny_state_outline.geoms) if hasattr(ny_state_outline, "geoms") else [ny_state_outline]
    ny_mask = np.zeros(Lon2d.size, dtype=bool)
    pts = np.vstack((Lon2d.ravel(), Lat2d.ravel())).T
    for poly in polys:
        coords = np.array(poly.exterior.coords)
        path = Path(coords)  # coords are (lon, lat)
        ny_mask |= path.contains_points(pts)
    ny_mask = ny_mask.reshape(Lon2d.shape)

    # Apply mask
    u2d = np.where(ny_mask, u2d, np.nan)
    v2d = np.where(ny_mask, v2d, np.nan)

    # Title/time calculation
    run_hour_map = {"00": 20, "06": 2, "12": 8, "18": 14}  # Map UTC run hour to local time
    base_time = datetime.strptime(f"{date_str} {hour_str}", "%Y%m%d %H")  # Base UTC time
    valid_time = base_time + timedelta(hours=step)  # Forecast valid time in UTC
    base_local_hour = run_hour_map.get(hour_str, int(hour_str))  # Local time for the run hour
    local_hour = (base_local_hour + (step - 1)) % 24  # Calculate local hour for the forecast step
    local_time = datetime.strptime(f"{local_hour:02d}", "%H").strftime("%I %p")  # Format as 12-hour time
    day_of_week = (base_time + timedelta(hours=step - 1)).strftime('%A')  # Adjust day if it crosses midnight

    title = (
        f"HRRR 10m Wind — New York (NY)\n"
        f"Valid: {valid_time.strftime('%Y-%m-%d %HZ')} ({day_of_week}, {local_time})  "
        f"Run: {hour_str}Z  Forecast Hour: {step}"
    )

    # Plotting setup
    fig = plt.figure(figsize=(10, 7), dpi=300, facecolor='white')
    fig.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.16)
    ax = plt.axes(projection=ccrs.PlateCarree(), facecolor='white')
    ax.set_extent(NY_EXTENT, crs=ccrs.PlateCarree())
    ax.set_title(title, fontsize=10, fontweight='bold', pad=10)

    # Base map
    ax.add_feature(cfeature.LAND, facecolor='white')
    ax.add_feature(cfeature.OCEAN, facecolor='white')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.7, edgecolor='black')
    ax.add_feature(cfeature.LAKES, facecolor='lightblue', edgecolor='black')

    # Wind barbs
    subsample_factor = 15  # Adjust this factor to control density
    Lon2d = Lon2d[::subsample_factor, ::subsample_factor]
    Lat2d = Lat2d[::subsample_factor, ::subsample_factor]
    u2d = u2d[::subsample_factor, ::subsample_factor]
    v2d = v2d[::subsample_factor, ::subsample_factor]
    ax.barbs(
        Lon2d, Lat2d, u2d, v2d,
        length=6, linewidth=1.0, color='red',  # Increased linewidth from 0.5 to 1.0
        transform=ccrs.PlateCarree(), zorder=2
    )

    # Overlay NY counties and state outline
    try:
        ny_gdf.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=0.5, zorder=7)
        gpd.GeoSeries([ny_state_outline], crs="EPSG:4326").boundary.plot(ax=ax, edgecolor="#000000", linewidth=1.0, zorder=8)
    except Exception as e:
        print(f"Error plotting NY overlays: {e}")

    # Add attribution text at the very bottom right of the map area, stacked tightly
    margin_x = (NY_EXTENT[1] - NY_EXTENT[0]) * 0.01
    margin_y = (NY_EXTENT[3] - NY_EXTENT[2]) * 0.01
    text_x = NY_EXTENT[1] - margin_x
    text_y_base = NY_EXTENT[2] + margin_y
    line_spacing = (NY_EXTENT[3] - NY_EXTENT[2]) * 0.025  # small vertical gap
    ax.text(
        text_x, text_y_base + line_spacing, "Images by Jack Fordyce",
        fontsize=7, color="black", ha="right", va="bottom",
        fontweight="normal", alpha=0.85,
        transform=ccrs.PlateCarree(),
        zorder=20,
        path_effects=[plt.matplotlib.patheffects.Stroke(linewidth=1, foreground='white'), plt.matplotlib.patheffects.Normal()]
    )
    ax.text(
        text_x, text_y_base, "Truelocalwx.com",
        fontsize=7, color="black", ha="right", va="bottom",
        fontweight="normal", alpha=0.85,
        transform=ccrs.PlateCarree(),
        zorder=20,
        path_effects=[plt.matplotlib.patheffects.Stroke(linewidth=1, foreground='white'), plt.matplotlib.patheffects.Normal()]
    )

    # Save PNG
    png_path = os.path.join(png_dir, f"hrrr_wind_10m_NY_{step:02d}.png")
    plt.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)

    # Explicitly delete large objects and collect garbage
    del ds_u, ds_v, u, v, Lon2d, Lat2d, u2d, v2d, fig, ax
    gc.collect()

    print(f"Generated PNG: {png_path}")
    return png_path

# Main process
for step in forecast_steps:
    wind_gribs = get_hrrr_grib([step], variables)
    for grib_files in wind_gribs:
        if grib_files:
            plot_wind_10m(grib_files['UGRD'], grib_files['VGRD'], step)

            # Explicitly delete processed GRIB files and collect garbage
            for grib_file in grib_files.values():
                if grib_file:
                    os.remove(grib_file)
            gc.collect()

print("HRRR 10m Wind processing complete.")
