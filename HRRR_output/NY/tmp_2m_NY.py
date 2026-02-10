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
import geopandas as gpd

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

# Set BASE_DIR relative to the script's location
BASE_DIR = os.path.join(os.path.dirname(__file__), "HRRR")

# Update directory structure
output_dir = os.path.join(BASE_DIR, "tmp_2m")
grib_dir = os.path.join(output_dir, "grib")
png_dir = os.path.join(output_dir, "png")

# Ensure the HRRR folder structure is created only if it doesn't already exist
if not os.path.exists(BASE_DIR):
    os.makedirs(grib_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)
    print(f"Created directory structure under: {BASE_DIR}")
else:
    print(f"Directory structure already exists under: {BASE_DIR}")

# HRRR URL and variables
base_url_hrrr = "https://nomads.ncep.noaa.gov/cgi-bin/filter_hrrr_2d.pl"
variable_tmp = "TMP"  # 2-meter temperature

# Calculate the most recent HRRR run dynamically
current_utc_time = datetime.utcnow()
available_hours = [0, 6, 12, 18]
most_recent_run_hour = max(h for h in available_hours if h <= current_utc_time.hour)
most_recent_run_time = current_utc_time.replace(hour=most_recent_run_hour, minute=0, second=0, microsecond=0)
previous_run_time = most_recent_run_time - timedelta(hours=6)

# Function to get date and hour strings for a given run time
def get_run_strings(run_time):
    return run_time.strftime("%Y%m%d"), run_time.strftime("%H")

# Start with the most recent run
date_str, hour_str = get_run_strings(most_recent_run_time)

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
forecast_steps = list(range(1, 49, 3))  # f01, f04, f07, ..., f46

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
        ds_tmp = xr.open_dataset(tmp_path, engine="cfgrib")
    except Exception as e:
        print(f"Error opening dataset: {e}")
        return None

    # Extract arrays
    tmp = ds_tmp.get('t2m')
    if tmp is None:
        print("Required variable not in dataset")
        return None
    tmp = (tmp.values - 273.15) * 9/5 + 32  # Convert from Kelvin to Fahrenheit

    lats = ds_tmp['latitude'].values
    lons = ds_tmp['longitude'].values
    lons_plot = np.where(lons > 180, lons - 360, lons)

    if lats.ndim == 1 and lons.ndim == 1:
        Lon2d, Lat2d = np.meshgrid(lons_plot, lats)
        tmp2d = tmp.squeeze()
    else:
        Lon2d, Lat2d = lons_plot, lats
        tmp2d = tmp.squeeze()

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
    tmp2d = np.where(ny_mask, tmp2d, np.nan)

    # Title/time calculation
    run_hour_map = {"00": 20, "06": 2, "12": 8, "18": 14}  # Map UTC run hour to local time
    base_time = datetime.strptime(f"{date_str} {hour_str}", "%Y%m%d %H")  # Base UTC time
    valid_time = base_time + timedelta(hours=step)  # Forecast valid time in UTC
    base_local_hour = run_hour_map.get(hour_str, int(hour_str))  # Local time for the run hour
    local_hour = (base_local_hour + (step - 1)) % 24  # Calculate local hour for the forecast step
    local_time = datetime.strptime(f"{local_hour:02d}", "%H").strftime("%I %p")  # Format as 12-hour time
    day_of_week = (base_time + timedelta(hours=step - 1)).strftime('%A')  # Adjust day if it crosses midnight

    title = (
        f"HRRR 2m Temperature — New York (NY)\n"
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

    # Colorbar
    cbar_ax = fig.add_axes([0.05, 0.08, 0.9, 0.02])  # Adjusted to match plot width
    cbar = plt.colorbar(mesh, cax=cbar_ax, orientation='horizontal')
    cbar.set_label("2m Temperature (°F)", fontsize=8)
    cbar.set_ticks(temp_levels)  # Ensure every tick is shown
    cbar.ax.tick_params(labelsize=6)

    # Overlay NY counties and state outline
    try:
        ny_gdf.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=0.5, zorder=7)
        gpd.GeoSeries([ny_state_outline], crs="EPSG:4326").boundary.plot(ax=ax, edgecolor="#000000", linewidth=1.0, zorder=8)
    except Exception as e:
        print(f"Error plotting NY overlays: {e}")

    # Save PNG
    png_path = os.path.join(png_dir, f"hrrr_tmp_2m_NY_{step:02d}.png")
    plt.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
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
    steps_to_process = range(step, step + 5)  # Process 3 steps at a time
    tmp_gribs = get_hrrr_grib(steps_to_process, "TMP")
    for i, tmp_grib in enumerate(tmp_gribs):
        if tmp_grib:
            plot_tmp_2m(tmp_grib, step + i)

print("HRRR 2m Temperature processing complete.")
