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
import time
import gc
from scipy import ndimage
from scipy.spatial import distance
from matplotlib import patheffects
from scipy.ndimage import maximum_filter, minimum_filter, label, generate_binary_structure
import geopandas as gpd
import shutil

# --- New: utility to fetch NY geojson and compute extent/boundary ---
def get_ny_geodata(padding_frac=0.09):  # Increased padding_frac from 0.05 to 0.2
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
grib_dir = os.path.join(BASE_DIR, "mslp_prate_csnow_NY", "grib")
png_dir = os.path.join(BASE_DIR, "mslp_prate_csnow_NY", "png")
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
processed_steps_file = os.path.join(BASE_DIR, "mslp_prate_csnow_NY", "processed_steps.txt")

# Clear the processed steps file at the start of a new run
if os.path.exists(processed_steps_file):
    os.remove(processed_steps_file)

# HRRR URL and variables
base_url_hrrr = "https://nomads.ncep.noaa.gov/cgi-bin/filter_hrrr_2d.pl"
variable_mslma = "MSLMA"  # Mean Sea Level Pressure
variable_prate = "PRATE"  # Precipitation Rate
variable_csnow = "CSNOW"  # Snowfall Rate
variable_cfrzr = "CFRZR"  # Freezing Rain Rate
variable_cicep = "CICEP"  # Sleet Rate

# Adjust available hours to include 03Z, 09Z, 15Z, 21Z with a max forecast range of 18 hours
available_hours = [0, 3, 6, 9, 12, 15, 18, 21]
max_forecast_hours = {3, 9, 15, 21}  # These runs only go out to 18 hours

# Function to get the forecast steps based on the run hour
def get_forecast_steps(run_hour):
    if run_hour in {3, 9, 15, 21}:  # Limit to 18-hour forecast for 03Z, 09Z, 15Z, 21Z
        return list(range(1, 19))  # 18-hour forecast
    return list(range(1, 49, 3))  # Default 48-hour forecast in 3-hour steps

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

# Levels and colormaps
mslp_levels = np.arange(960, 1050 + 2, 2)
prate_levels = [0.1, 0.25, 0.5, 0.75, 1.5, 2, 2.5, 3, 4, 6, 10, 16, 24]
prate_colors = [
    "#b6ffb6", "#54f354", "#19a319", "#016601", "#c9c938", "#f5f825",
    "#ffd700", "#ffa500", "#ff7f50", "#ff4500", "#ff1493", "#9400d3"
]
prate_cmap = LinearSegmentedColormap.from_list("prate_custom", prate_colors, N=len(prate_colors))
prate_norm = BoundaryNorm(prate_levels, prate_cmap.N)

# Forecast steps grouped into chunks of 24
forecast_steps = [list(range(i, i + 24)) for i in range(1, 49, 24)]  # [[1, 2, ..., 24], [25, 26, ..., 48]]

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

def get_hrrr_grib(step, variable):
    global date_str, hour_str  # Allow fallback to modify these variables
    file_name = f"hrrr.t{hour_str}z.wrfsfcf{step:02d}.grib2"
    file_path = os.path.join(grib_dir, f"{variable.lower()}_{file_name}")
    url = (
        f"{base_url_hrrr}?file={file_name}"
        f"&lev_surface=on&lev_mean_sea_level=on"
        f"&var_{variable}=on"
        f"&dir=%2Fhrrr.{date_str}%2Fconus"
    )
    if download_grib(url, file_path):
        return file_path
    else:
        # Fallback to the previous run if the most recent run fails
        print(f"Falling back to previous run for step {step}, variable {variable}")
        fallback_date_str, fallback_hour_str = get_run_strings(previous_run_time)
        fallback_file_name = f"hrrr.t{fallback_hour_str}z.wrfsfcf{step:02d}.grib2"
        fallback_file_path = os.path.join(grib_dir, f"{variable.lower()}_{fallback_file_name}")
        fallback_url = (
            f"{base_url_hrrr}?file={fallback_file_name}"
            f"&lev_surface=on&lev_mean_sea_level=on"
            f"&var_{variable}=on"
            f"&dir=%2Fhrrr.{fallback_date_str}%2Fconus"
        )
        if download_grib(fallback_url, fallback_file_path):
            # Update global date_str and hour_str to fallback values
            date_str, hour_str = fallback_date_str, fallback_hour_str
            return fallback_file_path
        else:
            print(f"Failed to download {variable} for both runs (step {step})")
            return None

# --- Plotting function (uses adjusted NY extent) ---
def plot_combined(mslp_path, prate_path, step, csnow_path=None, cfrzr_path=None, cicep_path=None):
    try:
        # Open datasets
        ds_mslp = xr.open_dataset(mslp_path, engine="cfgrib")
        ds_prate = xr.open_dataset(prate_path, engine="cfgrib")
        ds_csnow = xr.open_dataset(csnow_path, engine="cfgrib") if csnow_path else None
        ds_cfrzr = xr.open_dataset(cfrzr_path, engine="cfgrib") if cfrzr_path else None
        ds_cicep = xr.open_dataset(cicep_path, engine="cfgrib") if cicep_path else None

        # extract arrays
        mslp = ds_mslp.get('mslma')
        prate = ds_prate.get('prate')
        if mslp is None or prate is None:
            print("Required variables not in datasets")
            return None
        mslp = mslp.values / 100.0  # Pa to hPa
        prate = prate.values * 3600  # mm/s to mm/hr

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

        # --- Ensure rate arrays exist before masking (prevent UnboundLocalError) ---
        snow_rate2d = None
        cfrzr_rate2d = None
        cicep_rate2d = None

        # compute snow_rate2d if csnow dataset present
        if ds_csnow is not None and "csnow" in ds_csnow:
            try:
                csnow = ds_csnow['csnow'].values * 3600
                csnow2d = csnow.squeeze()
                if csnow2d.shape == prate2d.shape:
                    snow_mask = (csnow2d > 0)
                    snow_rate2d = np.where(snow_mask, prate2d, np.nan)
            except Exception:
                snow_rate2d = None

        # compute cfrzr_rate2d if cfrzr dataset present
        if ds_cfrzr is not None and "cfrzr" in ds_cfrzr:
            try:
                cfrzr = ds_cfrzr['cfrzr'].values * 3600
                cfrzr2d = cfrzr.squeeze()
                if cfrzr2d.shape == prate2d.shape:
                    cfrzr_mask = (cfrzr2d > 0)
                    cfrzr_rate2d = np.where(cfrzr_mask, prate2d, np.nan)
            except Exception:
                cfrzr_rate2d = None

        # compute cicep_rate2d if cicep dataset present
        if ds_cicep is not None and "cicep" in ds_cicep:
            try:
                cicep = ds_cicep['cicep'].values * 3600
                cicep2d = cicep.squeeze()
                if cicep2d.shape == prate2d.shape:
                    cicep_mask = (cicep2d > 0)
                    cicep_rate2d = np.where(cicep_mask, prate2d, np.nan)
            except Exception:
                cicep_rate2d = None

        # --- Mask all fields to New York polygon so only NY appears ---
        from matplotlib.path import Path
        polys = list(ny_state_outline.geoms) if hasattr(ny_state_outline, "geoms") else [ny_state_outline]
        ny_mask = np.zeros(Lon2d.size, dtype=bool)
        pts = np.vstack((Lon2d.ravel(), Lat2d.ravel())).T
        for poly in polys:
            coords = np.array(poly.exterior.coords)
            path = Path(coords)  # coords are (lon, lat)
            ny_mask |= path.contains_points(pts)
        ny_mask = ny_mask.reshape(Lon2d.shape)

        # apply mask to primary plotted arrays
        prate2d = np.where(ny_mask, prate2d, np.nan)
        mslp2d = np.where(ny_mask, mslp2d, np.nan)
        if snow_rate2d is not None:
            snow_rate2d = np.where(ny_mask, snow_rate2d, np.nan)
        if cfrzr_rate2d is not None:
            cfrzr_rate2d = np.where(ny_mask, cfrzr_rate2d, np.nan)
        if cicep_rate2d is not None:
            cicep_rate2d = np.where(ny_mask, cicep_rate2d, np.nan)

        # --- Title/time calculation --- (replace with clearer NY-specific title)
        run_hour_map = {"00": 20, "06": 2, "12": 8, "18": 14}  # Map UTC run hour to local time
        base_time = datetime.strptime(f"{date_str} {hour_str}", "%Y%m%d %H")  # Base UTC time
        valid_time = base_time + timedelta(hours=step)  # Forecast valid time in UTC
        base_local_hour = run_hour_map.get(hour_str, int(hour_str))  # Local time for the run hour
        local_hour = (base_local_hour + (step - 1)) % 24  # Calculate local hour for the forecast step
        local_time = datetime.strptime(f"{local_hour:02d}", "%H").strftime("%I %p")  # Format as 12-hour time
        day_of_week = (base_time + timedelta(hours=step - 1)).strftime('%A')  # Adjust day if it crosses midnight

        title = (
            f"HRRR Precipitation Rate & Mean Sea Level Pressure — New York (NY)\n"
            f"Valid: {valid_time.strftime('%Y-%m-%d %HZ')} ({day_of_week}, {local_time})  "
            f"Run: {hour_str}Z  Forecast Hour: {step}"
        )

        # --- Plotting setup (use adjusted NY extent) ---
        fig = plt.figure(figsize=(10, 7), dpi=300, facecolor='white')
        fig.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.16)
        ax = plt.axes(projection=ccrs.PlateCarree(), facecolor='white')

        # Adjust the extent to zoom out slightly
        zoom_factor = 0.9  # Increased zoom_factor to zoom out slightly
        extent = [
            NY_EXTENT[0] + (NY_EXTENT[1] - NY_EXTENT[0]) * (1 - zoom_factor) / 2,
            NY_EXTENT[1] - (NY_EXTENT[1] - NY_EXTENT[0]) * (1 - zoom_factor) / 2,
            NY_EXTENT[2] + (NY_EXTENT[3] - NY_EXTENT[2]) * (1 - zoom_factor) / 2,
            NY_EXTENT[3] - (NY_EXTENT[3] - NY_EXTENT[2]) * (1 - zoom_factor) / 2,
        ]
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.set_title(title, fontsize=10, fontweight='bold', pad=10)

        # Base map limited to NY: make anything outside appear white
        ax.add_feature(cfeature.LAND, facecolor='white')
        ax.add_feature(cfeature.OCEAN, facecolor='white')
        ax.add_feature(cfeature.COASTLINE, linewidth=0.7, edgecolor='black')  # Added coastlines
        ax.add_feature(cfeature.LAKES, facecolor='lightblue', edgecolor='black')  # Added lakes
        ax.set_facecolor('white')  # ensure background is white

        # PRATE plotting
        mesh = ax.contourf(
            Lon2d, Lat2d, prate2d,
            levels=prate_levels,
            cmap=prate_cmap,
            norm=prate_norm,
            extend='max',
            transform=ccrs.PlateCarree(),
            alpha=0.8,
            zorder=1
        )

        # CFRZR, CICEP, CSNOW plotting (if present)
        if cfrzr_rate2d is not None:
            cfrzr_levels = [0.1, 0.25, 0.5, 1, 2, 4, 8, 16]
            cfrzr_colors = ["#fce4ec", "#f8bbd0", "#f48fb1", "#ec407a", "#d81b60", "#880e4f", "#560027"]
            cfrzr_cmap = LinearSegmentedColormap.from_list("cfrzr_cbar", cfrzr_colors, N=len(cfrzr_colors))
            cfrzr_norm = BoundaryNorm(cfrzr_levels, cfrzr_cmap.N)
            cfrzr_mesh = ax.contourf(Lon2d, Lat2d, cfrzr_rate2d, levels=cfrzr_levels, cmap=cfrzr_cmap, norm=cfrzr_norm, extend='max', transform=ccrs.PlateCarree(), alpha=0.85, zorder=3)

        if cicep_rate2d is not None:
            cicep_levels = [0.1, 0.25, 0.5, 1, 2, 4, 8, 16]
            cicep_colors = ["#f3e5f5", "#e1bee7", "#ce93d8", "#ba68c8", "#9c27b0", "#7b1fa2", "#4a148c", "#12005e"]
            cicep_cmap = LinearSegmentedColormap.from_list("cicep_cbar", cicep_colors, N=len(cicep_colors))
            cicep_norm = BoundaryNorm(cicep_levels, cicep_cmap.N)
            cicep_mesh = ax.contourf(Lon2d, Lat2d, cicep_rate2d, levels=cicep_levels, cmap=cicep_cmap, norm=cicep_norm, extend='max', transform=ccrs.PlateCarree(), alpha=0.85, zorder=3)

        # MSLP plotting (contours)
        cs = ax.contour(Lon2d, Lat2d, mslp2d, levels=mslp_levels, colors='black', linewidths=0.7, transform=ccrs.PlateCarree(), zorder=4)
        ax.clabel(cs, fmt='%d', fontsize=6, colors='black', inline=True)

        # Snow plotting
        if snow_rate2d is not None:
            snow_levels = [0.1, 0.25, 0.5, 1, 2, 4, 8, 16]
            snow_colors = ["#e3f2fd", "#bbdefb", "#90caf9", "#42a5f5", "#1e88e5", "#1565c0", "#0d47a1", "#002171"]
            snow_cmap = LinearSegmentedColormap.from_list("snow_cbar", snow_colors, N=len(snow_colors))
            snow_norm = BoundaryNorm(snow_levels, snow_cmap.N)
            snow_mesh = ax.contourf(Lon2d, Lat2d, snow_rate2d, levels=snow_levels, cmap=snow_cmap, norm=snow_norm, extend='max', transform=ccrs.PlateCarree(), alpha=0.85, zorder=3)

        # Colorbars placement — create four identical smaller axes at bottom; leave blank if mesh missing
        cbar_y = 0.055
        cbar_h = 0.02   # smaller height
        cbar_w = 0.21   # width chosen so 4 bars + gaps fit within [0.05,0.95]
        gap = 0.02
        x0 = 0.05
        x1 = x0 + cbar_w + gap
        x2 = x1 + cbar_w + gap
        x3 = x2 + cbar_w + gap
        cax_prate = fig.add_axes([x0, cbar_y, cbar_w, cbar_h])
        cax_cfrzr = fig.add_axes([x1, cbar_y, cbar_w, cbar_h])
        cax_csnow = fig.add_axes([x2, cbar_y, cbar_w, cbar_h])
        cax_cicep = fig.add_axes([x3, cbar_y, cbar_w, cbar_h])

        # Ensure axes have white background and hide ticks for empty axes
        for cax in (cax_prate, cax_cfrzr, cax_csnow, cax_cicep):
            cax.set_facecolor('white')
            cax.tick_params(labelbottom=False, bottom=False)
            cax.set_xticks([])

        # helper to remove leading zero from decimals (e.g. "0.25" -> ".25")
        def fmt_tick(v):
            s = f"{v:g}"
            if s.startswith("-0."):
                return "-" + s[2:]
            if s.startswith("0."):
                return s[1:]
            return s

        # PRATE colorbar (always present)
        cbar = plt.colorbar(mesh, cax=cax_prate, orientation='horizontal', ticks=prate_levels[::2], boundaries=prate_levels)
        prate_tick_labels = [fmt_tick(v) for v in prate_levels[::2]]
        cbar.ax.set_xticklabels(prate_tick_labels)
        cbar.set_label("Precipitation Rate (mm/hr)", fontsize=7, labelpad=2)  # slightly smaller
        cbar.ax.tick_params(labelsize=6, length=1)
        cbar.ax.set_facecolor('white')
        cbar.outline.set_edgecolor('black')

        # CFRZR colorbar (only if plotted)
        if cfrzr_rate2d is not None:
            cbar_cfrzr = plt.colorbar(cfrzr_mesh, cax=cax_cfrzr, orientation='horizontal', ticks=cfrzr_levels, boundaries=cfrzr_levels)
            cbar_cfrzr.ax.set_xticklabels([fmt_tick(v) for v in cfrzr_levels])
            cbar_cfrzr.ax.tick_params(labelsize=6)
            cbar_cfrzr.set_label("Freezing Rain (mm/hr)", fontsize=6)
        else:
            cax_cfrzr.set_axis_off()

        # CSNOW colorbar (only if plotted)
        if snow_rate2d is not None:
            cbar_csnow = plt.colorbar(snow_mesh, cax=cax_csnow, orientation='horizontal', ticks=snow_levels, boundaries=snow_levels)
            cbar_csnow.ax.set_xticklabels([fmt_tick(v) for v in snow_levels])
            cbar_csnow.ax.tick_params(labelsize=6)
            cbar_csnow.set_label("Snow Rate (mm/hr)", fontsize=6)
        else:
            cax_csnow.set_axis_off()

        # CICEP colorbar (only if plotted)
        if cicep_rate2d is not None:
            cbar_cicep = plt.colorbar(cicep_mesh, cax=cax_cicep, orientation='horizontal', ticks=cicep_levels, boundaries=cicep_levels)
            cbar_cicep.ax.set_xticklabels([fmt_tick(v) for v in cicep_levels])
            cbar_cicep.ax.tick_params(labelsize=6)
            cbar_cicep.set_label("Sleet (mm/hr)", fontsize=6)
        else:
            cax_cicep.set_axis_off()

        # Detect highs and lows limited to NY extent
        mask = (
            (Lon2d >= extent[0]) & (Lon2d <= extent[1]) &
            (Lat2d >= extent[2]) & (Lat2d <= extent[3])
        )
        data_masked = np.where(mask, mslp2d, np.nan)

        def find_valid_extrema(extrema_y, extrema_x, values, margin, is_high=True):
            sorted_indices = np.argsort(values)[::-1] if is_high else np.argsort(values)
            for idx in sorted_indices:
                y, x = extrema_y[idx], extrema_x[idx]
                lon, lat = Lon2d[y, x], Lat2d[y, x]
                if (extent[0] + margin <= lon <= extent[1] - margin) and (extent[2] + margin <= lat <= extent[3] - margin):
                    return lon, lat, values[idx]
            return None, None, None

        # Plot one low if <= 1005 hPa
        min_filt = ndimage.minimum_filter(data_masked, size=25, mode='constant', cval=np.nan)
        lows = (data_masked == min_filt) & ~np.isnan(data_masked)
        low_y, low_x = np.where(lows)
        low_values = data_masked[low_y, low_x]

        low_lon, low_lat, low_value = find_valid_extrema(low_y, low_x, low_values, margin=0.1, is_high=False)
        if low_lon is not None and low_value <= 1005:  # Only plot if <= 1005 hPa
            ax.text(low_lon, low_lat, "L", color='red', fontsize=12, fontweight='bold', ha='center', va='center', transform=ccrs.PlateCarree(), zorder=6, path_effects=[patheffects.Stroke(linewidth=1, foreground='white'), patheffects.Normal()])
            ax.text(low_lon, low_lat - 0.2, f"{low_value:.0f}", color='red', fontsize=6, fontweight='bold', ha='center', va='top', transform=ccrs.PlateCarree(), zorder=6)

        # Plot one high if > 1029 hPa
        max_filt = ndimage.maximum_filter(data_masked, size=25, mode='constant', cval=np.nan)
        highs = (data_masked == max_filt) & ~np.isnan(data_masked)
        high_y, high_x = np.where(highs)
        high_values = data_masked[high_y, high_x]

        high_lon, high_lat, high_value = find_valid_extrema(high_y, high_x, high_values, margin=0.1, is_high=True)
        if high_lon is not None and high_value > 1029:  # Only plot if > 1029 hPa
            ax.text(high_lon, high_lat, "H", color='blue', fontsize=12, fontweight='bold', ha='center', va='center', transform=ccrs.PlateCarree(), zorder=6, path_effects=[patheffects.Stroke(linewidth=1, foreground='white'), patheffects.Normal()])
            ax.text(high_lon, high_lat - 0.2, f"{high_value:.0f}", color='blue', fontsize=6, fontweight='bold', ha='center', va='top', transform=ccrs.PlateCarree(), zorder=6)


        # Overlay NY counties and state outline to ensure PNGs clearly show NY only
        try:
            ny_gdf.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=0.5, zorder=7)
            gpd.GeoSeries([ny_state_outline], crs="EPSG:4326").boundary.plot(ax=ax, edgecolor="#000000", linewidth=1.0, zorder=8)
        except Exception as e:
            print(f"Error plotting NY overlays: {e}")

        # Move attribution text further to the right and up from the bottom left
        margin_x = (NY_EXTENT[1] - NY_EXTENT[0]) * 0.06  # even more right
        margin_y = (NY_EXTENT[3] - NY_EXTENT[2]) * 0.075  # even more up
        text_x = NY_EXTENT[0] + margin_x
        text_y_base = NY_EXTENT[2] + margin_y
        line_spacing = (NY_EXTENT[3] - NY_EXTENT[2]) * 0.03  # vertical gap
        ax.text(
            text_x, text_y_base + line_spacing, "Images by Jack Fordyce",
            fontsize=7, color="black", ha="left", va="bottom",
            fontweight="normal", alpha=0.85,
            transform=ccrs.PlateCarree(),
            zorder=20,
            path_effects=[patheffects.Stroke(linewidth=1, foreground='white'), patheffects.Normal()]
        )
        ax.text(
            text_x, text_y_base, "Truelocalwx.com",
            fontsize=7, color="black", ha="left", va="bottom",
            fontweight="normal", alpha=0.85,
            transform=ccrs.PlateCarree(),
            zorder=20,
            path_effects=[patheffects.Stroke(linewidth=1, foreground='white'), patheffects.Normal()]
        )

        # Save only NY PNGs
        png_path = os.path.join(png_dir, f"hrrr_combined_NY_{step:02d}.png")
        plt.savefig(png_path, bbox_inches="tight", dpi=300)
        plt.close(fig)
        print(f"Generated PNG: {png_path}")
        return png_path

    except Exception as e:
        print(f"Error in plot_combined: {e}")

# Function to clear all files in a directory
def clear_folder(folder_path):
    if os.path.exists(folder_path):
        for f in os.listdir(folder_path):
            file_path = os.path.join(folder_path, f)
            if os.path.isfile(file_path):
                os.remove(file_path)
        print(f"Cleared folder: {folder_path}")

# Clear the folders at the start of the script
clear_folder(grib_dir)
clear_folder(png_dir)

# Main process
for step_group in forecast_steps:
    for step in step_group:
        mslp_grib = get_hrrr_grib(step, "MSLMA")
        prate_grib = get_hrrr_grib(step, "PRATE")
        csnow_grib = get_hrrr_grib(step, "CSNOW")
        cfrzr_grib = get_hrrr_grib(step, "CFRZR")
        cicep_grib = get_hrrr_grib(step, "CICEP")

        if mslp_grib and prate_grib:
            plot_combined(mslp_grib, prate_grib, step, csnow_grib, cfrzr_grib, cicep_grib)

            # Delete GRIB files after processing
            for grib_file in [mslp_grib, prate_grib, csnow_grib, cfrzr_grib, cicep_grib]:
                if grib_file and os.path.exists(grib_file):
                    os.remove(grib_file)

            # Collect garbage
            gc.collect()

print("HRRR processing complete.")
