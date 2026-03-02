def list_pngs(png_dir):
    """Return a list of all PNG files in the given directory (with full paths)."""
    return [os.path.join(png_dir, f) for f in os.listdir(png_dir) if f.lower().endswith('.png') and os.path.isfile(os.path.join(png_dir, f))]
import shutil
import os
from datetime import datetime, timedelta

def get_save_dirs(base_dir, run_datetime):
    """
    Create and return GRIB and PNG output directories in the format:
    /var/data/yymmdd/run_hour/grib
    /var/data/yymmdd/run_hour/png
    """
    # Subtract one hour to get the most recent completed run hour
    run_time = run_datetime - timedelta(hours=1)
    yymmdd = run_time.strftime('%y%m%d')
    run_hour = run_time.strftime('%H') + 'Z'  # Format as '03Z', '09Z', etc.
    root_dir = os.path.join(base_dir, "mslp_prate_csnow_EAST", yymmdd, run_hour)
    grib_dir = os.path.join(root_dir, 'grib')
    png_dir = os.path.join(root_dir, 'png')
    os.makedirs(grib_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)
    return grib_dir, png_dir

# --- HRRR run hour logic ---
def find_most_recent_hrrr_run(current_utc_time=None, available_hours=None, is_valid_run=None):
    """
    Mimics the HRRR logic to find the most recent valid run hour.
    - current_utc_time: datetime object (default: now)
    - available_hours: list of valid run hours (default: 0-23)
    - is_valid_run: function(run_time) -> bool (default: always True)
    Returns: datetime object for the most recent valid run
    """
    if current_utc_time is None:
        current_utc_time = datetime.utcnow()
    if available_hours is None:
        available_hours = list(range(24))
    if is_valid_run is None:
        # By default, assume all hours are valid (for demonstration)
        def is_valid_run(rt):
            return True
    most_recent_run_time = None
    for offset in range(24):
        candidate_hour = (current_utc_time.hour - offset) % 24
        if candidate_hour in available_hours:
            candidate_time = current_utc_time.replace(hour=candidate_hour, minute=0, second=0, microsecond=0)
            if is_valid_run(candidate_time):
                most_recent_run_time = candidate_time
                break
    if most_recent_run_time is None:
        # Fallback: just use current time
        most_recent_run_time = current_utc_time.replace(minute=0, second=0, microsecond=0)
    return most_recent_run_time

# --- Copy PNGs from flat dir to dated dir ---
def copy_pngs_to_dated_dir(flat_png_dir, dated_png_dir):
    """
    Copy all PNG files from flat_png_dir to dated_png_dir.
    After copying, keep only the 7 most recent dated run directories (by creation time), deleting the oldest if there are more than 7.
    """
    if not os.path.exists(flat_png_dir):
        print(f"Source PNG directory does not exist: {flat_png_dir}")
        return
    os.makedirs(dated_png_dir, exist_ok=True)
    for png_file in os.listdir(flat_png_dir):
        if png_file.lower().endswith('.png'):
            src = os.path.join(flat_png_dir, png_file)
            dst = os.path.join(dated_png_dir, png_file)
            shutil.copy2(src, dst)
            print(f"Copied {src} -> {dst}")

    # --- Keep only 7 most recent dated run directories ---
    # The parent of dated_png_dir is the run_hour dir, its parent is the yymmdd dir, its parent is mslp_prate_csnow_EAST
    # We want to keep only the 7 most recent run_hour dirs under each yymmdd dir
    yymmdd_dir = os.path.dirname(os.path.dirname(dated_png_dir))
    for day_dir in [yymmdd_dir]:
        if os.path.exists(day_dir):
            run_hour_dirs = [os.path.join(day_dir, d) for d in os.listdir(day_dir) if os.path.isdir(os.path.join(day_dir, d))]
            # Sort by directory creation time (oldest first)
            run_hour_dirs.sort(key=lambda d: os.path.getctime(d))
            if len(run_hour_dirs) > 7:
                dirs_to_delete = run_hour_dirs[:-7]
                for old_dir in dirs_to_delete:
                    try:
                        shutil.rmtree(old_dir)
                        print(f"Deleted old run directory: {old_dir}")
                    except Exception as e:
                        print(f"Error deleting {old_dir}: {e}")

if __name__ == "__main__":
    # Example usage
    BASE_DIR = '/var/data'
    # HRRR available hours
    available_hours = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
    # Placeholder: all hours are valid (replace with real check if needed)
    run_datetime = find_most_recent_hrrr_run(datetime.utcnow(), available_hours)
    grib_dir, png_dir = get_save_dirs(BASE_DIR, run_datetime)
    print(f"GRIB directory: {grib_dir}")
    print(f"PNG directory: {png_dir}")
    png_files = list_pngs(png_dir)
    print("PNG files:", png_files)

    # Copy PNGs from flat dir to dated dir
    flat_png_dir = os.path.join(BASE_DIR, "mslp_prate_csnow_EAST", "png")
    copy_pngs_to_dated_dir(flat_png_dir, png_dir)
