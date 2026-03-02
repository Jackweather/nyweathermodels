import os
import shutil

import requests
from datetime import datetime

# List of subfolders for each variable (must match the output folders in /var/data)
PNG_SUBFOLDERS = [
    "tmp_2m_EAST/png",
    "total_precip_EAST/png",
    "SNOW_10to_1_EAST/png",
    "snow_8_to_1_EAST/png",
    "wind_10m_EAST/png",
    "vis_EAST/png",
]


BASE_DIR = "/var/data"

# HRRR URL and available hours (copied from forecast scripts)
base_url_hrrr = "https://nomads.ncep.noaa.gov/cgi-bin/filter_hrrr_2d.pl"
available_hours = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

# Function to validate if a run time is accessible (HEAD request)
def is_valid_run(run_time):
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


def main():
    # Find the most recent valid HRRR run (UTC)
    current_utc_time = datetime.utcnow()
    most_recent_run_time = None
    for offset in range(24):  # Check up to 24 hours back
        candidate_hour = (current_utc_time.hour - offset) % 24
        if candidate_hour in available_hours:
            candidate_time = current_utc_time.replace(hour=candidate_hour, minute=0, second=0, microsecond=0)
            if is_valid_run(candidate_time):
                most_recent_run_time = candidate_time
                break
    if most_recent_run_time is None:
        print("No valid HRRR run found in the last 24 hours. Using current UTC time.")
        most_recent_run_time = current_utc_time

    yymmdd = most_recent_run_time.strftime("%y%m%d")
    run_hour = most_recent_run_time.strftime("%H")
    target_dir = os.path.join(BASE_DIR, yymmdd, run_hour)
    os.makedirs(target_dir, exist_ok=True)


    for subfolder in PNG_SUBFOLDERS:
        png_dir = os.path.join(BASE_DIR, subfolder)
        if not os.path.exists(png_dir):
            continue
        # Get a clean prefix from the subfolder (e.g., tmp_2m_EAST)
        prefix = os.path.basename(os.path.dirname(png_dir))
        for fname in os.listdir(png_dir):
            if fname.lower().endswith(".png"):
                src = os.path.join(png_dir, fname)
                dst_name = f"{prefix}_{fname}"
                dst = os.path.join(target_dir, dst_name)
                shutil.copy2(src, dst)
                print(f"Copied {src} -> {dst}")

    print(f"All PNGs saved to {target_dir}")

if __name__ == "__main__":
    main()
