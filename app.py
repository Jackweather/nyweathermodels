from flask import Flask, jsonify, send_from_directory, render_template
import os
import subprocess
import threading
import getpass

app = Flask(__name__, static_folder="templates", static_url_path="/static")

# --- New Endpoints for dynamic date/run hour selection ---
@app.route("/get_all_dated_dirs")
def get_all_dated_dirs():
    base = os.path.join(BASE_DIR, "mslp_prate_csnow_EAST")
    try:
        if not os.path.exists(base):
            return jsonify([])
        yymmdd_dirs = [d for d in os.listdir(base) if d.isdigit() and len(d) == 6 and os.path.isdir(os.path.join(base, d))]
        return jsonify(sorted(yymmdd_dirs))
    except Exception as e:
        return jsonify([])

@app.route("/get_run_hours/<yymmdd>")
def get_run_hours(yymmdd):
    base = os.path.join(BASE_DIR, "mslp_prate_csnow_EAST", yymmdd)
    try:
        if not os.path.exists(base):
            return jsonify([])
        run_hours = [d for d in os.listdir(base) if d.endswith('Z') and len(d) == 3 and os.path.isdir(os.path.join(base, d))]
        return jsonify(sorted(run_hours))
    except Exception as e:
        return jsonify([])

BASE_DIR = "/var/data"
PNG_DIRS = {
    "mslp": os.path.join(BASE_DIR, "mslp_prate_csnow_EAST", "png"),
    "temp": os.path.join(BASE_DIR, "tmp_2m_EAST", "png"),
    "snow_8_to_1": os.path.join(BASE_DIR, "snow_8_to_1_EAST", "png"),
    "snow_10_to_1": os.path.join(BASE_DIR, "SNOW_10to_1_EAST", "png"),
    "precip_type_rate": os.path.join(BASE_DIR, "mslp_prate_csnow_EAST", "png"),
    "wind": os.path.join(BASE_DIR, "wind_10m_EAST", "png"),
    "total_precip": os.path.join(BASE_DIR, "total_precip_EAST", "png"),
    "visibility": os.path.join(BASE_DIR, "vis_EAST", "png"),
    "SNOW_Depth": os.path.join(BASE_DIR, "SNOD_EAST", "png"),
}


# --- Utility Functions ---
def list_png_files(view):
    png_dir = PNG_DIRS.get(view)
    if not png_dir:
        return jsonify({"error": "Invalid view"}), 404
    if view == "gfs":
        png_files = sorted([f"/get_gfs_png/{f}" for f in os.listdir(png_dir) if f.endswith(".png")])
    else:
        png_files = sorted([f"/get_png/{view}/{f}" for f in os.listdir(png_dir) if f.endswith(".png")])
    return jsonify(png_files)

def serve_png_file(view, filename):
    png_dir = PNG_DIRS.get(view)
    if not png_dir:
        return jsonify({"error": "Invalid view"}), 404
    return send_from_directory(png_dir, filename)

def run_scripts(scripts, max_concurrent):
    semaphore = threading.Semaphore(max_concurrent)
    def run_script(script, cwd):
        with semaphore:
            try:
                with subprocess.Popen(
                    ["python", script],
                    cwd=cwd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                ) as process:
                    for line in process.stdout:
                        print(line, end="")
                    for line in process.stderr:
                        print(line, end="")
                    process.wait()
                    if process.returncode != 0:
                        print(f"Error: {os.path.basename(script)} exited with code {process.returncode}")
                    else:
                        print(f"{os.path.basename(script)} ran successfully!")
            except Exception as e:
                print(f"Error running {os.path.basename(script)}: {e}")
    threads = [threading.Thread(target=run_script, args=(script, cwd)) for script, cwd in scripts]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

# --- Routes ---
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/saved_view")
def saved_view():
    return render_template("saved.html")

@app.route("/list_dated_pngs/<yymmdd>/<run_hour>")
def list_dated_pngs(yymmdd, run_hour):
    png_dir = os.path.join(BASE_DIR, "mslp_prate_csnow_EAST", yymmdd, run_hour, "png")
    if not os.path.exists(png_dir):
        return jsonify([])
    files = [f for f in os.listdir(png_dir) if f.lower().endswith('.png')]
    return jsonify(sorted(files))

@app.route("/get_latest_dated_dir")
def get_latest_dated_dir():
    base = os.path.join(BASE_DIR, "mslp_prate_csnow_EAST")
    try:
        yymmdd_dirs = [d for d in os.listdir(base) if d.isdigit() and len(d) == 6]
        if not yymmdd_dirs:
            return jsonify({"yymmdd": None, "run_hour": None})
        latest_yymmdd = max(yymmdd_dirs)
        run_dir = os.path.join(base, latest_yymmdd)
        run_hours = [d for d in os.listdir(run_dir) if d.endswith('Z') and len(d) == 3]
        if not run_hours:
            return jsonify({"yymmdd": latest_yymmdd, "run_hour": None})
        latest_run_hour = max(run_hours)
        png_dir = os.path.join(run_dir, latest_run_hour, "png")
        has_pngs = os.path.exists(png_dir) and any(f.lower().endswith('.png') for f in os.listdir(png_dir))
        return jsonify({"yymmdd": latest_yymmdd, "run_hour": latest_run_hour, "has_pngs": has_pngs})
    except Exception as e:
        return jsonify({"yymmdd": None, "run_hour": None, "error": str(e)})

@app.route("/dated_pngs/<yymmdd>/<run_hour>/<filename>")
def serve_dated_png(yymmdd, run_hour, filename):
    png_dir = os.path.join(BASE_DIR, "mslp_prate_csnow_EAST", yymmdd, run_hour, "png")
    if not os.path.exists(os.path.join(png_dir, filename)):
        return jsonify({"error": "File not found"}), 404
    return send_from_directory(png_dir, filename)

@app.route("/list_pngs/<view>")
def list_pngs(view):
    return list_png_files(view)

@app.route("/get_png/<view>/<filename>")
def get_png(view, filename):
    return serve_png_file(view, filename)

@app.route('/assets/<path:filename>')
def serve_asset(filename):
    assets_dir = os.path.join(os.path.dirname(__file__), 'assets')
    return send_from_directory(assets_dir, filename)

@app.route("/run-task1")
def run_task1():
    scripts = [
        ("/opt/render/project/src/NY/mslp_prate_csnow_EAST.py", "/opt/render/project/src/NY"),
        ("/opt/render/project/src/NY/tmp_2m_EAST.py", "/opt/render/project/src/NY"),
        ("/opt/render/project/src/NY/Snow_liquid_ratio_8to1_EAST.py", "/opt/render/project/src/NY"),
        ("/opt/render/project/src/NY/Snow_liquid_ratio_10to1_EAST.py", "/opt/render/project/src/NY"),
        ("/opt/render/project/src/NY/wind_10m_EAST.py", "/opt/render/project/src/NY"),
        ("/opt/render/project/src/NY/total_precip_EAST.py", "/opt/render/project/src/NY"),
        ("/opt/render/project/src/NY/vis_EAST.py", "/opt/render/project/src/NY"),
        ("/opt/render/project/src/NY/Snow_liquid_ratio_SNOD_EAST.py", "/opt/render/project/src/NY"),
        
    ]
    threading.Thread(target=lambda: run_scripts(scripts, 3)).start()
    return "Task started in background! Check logs folder for output.", 200

@app.route("/run-task2")
def run_task2():
    scripts = [
        ("/opt/render/project/src/GFS_USA/Prate_USA.py", "/opt/render/project/src/GFS_USA"),
        # Add more GFS scripts here as needed
    ]
    threading.Thread(target=lambda: run_scripts(scripts, 2)).start()
    return "Task 2 started in background! Check logs folder for output.", 200

if __name__ == "__main__":
    app.run(debug=True)
