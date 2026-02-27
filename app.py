from flask import Flask, jsonify, send_from_directory, render_template
import os
import subprocess
import traceback
import threading
import getpass

app = Flask(__name__, static_folder="templates", static_url_path="/static")

# Directories containing the PNG files
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
    "gfs_precip": os.path.join(BASE_DIR, "GSF_mslp_prate_csnow", "png"),
    "gfs_tmp": os.path.join(BASE_DIR, "GFS_tmp_2m", "png"),
    "850_vort": os.path.join(BASE_DIR, "GFS_absv_850mb", "png"),
}

@app.route("/get_gfs_png/<filename>")
def get_gfs_png(filename):
    # Serve GFS PNGs from the correct directory
    gfs_dir = PNG_DIRS.get("gfs")
    if not gfs_dir:
        return jsonify({"error": "GFS directory not found"}), 404
    return send_from_directory(gfs_dir, filename)

@app.route("/")
def index():
    # Render the HTML frontend
    return render_template("index.html")

def list_png_files(view):
    # Helper function to list PNG files in a directory
    png_dir = PNG_DIRS.get(view)
    if not png_dir:
        return jsonify({"error": "Invalid view"}), 404
    if view == "gfs":
        # For GFS, use the /get_gfs_png endpoint for URLs
        png_files = sorted([f"/get_gfs_png/{f}" for f in os.listdir(png_dir) if f.endswith(".png")])
    else:
        png_files = sorted([f"/get_png/{view}/{f}" for f in os.listdir(png_dir) if f.endswith(".png")])
    return jsonify(png_files)

def serve_png_file(view, filename):
    # Helper function to serve a specific PNG file
    png_dir = PNG_DIRS.get(view)
    if not png_dir:
        return jsonify({"error": "Invalid view"}), 404
    return send_from_directory(png_dir, filename)

@app.route("/list_pngs/<view>")
def list_pngs(view):
    return list_png_files(view)

@app.route("/get_png/<view>/<filename>")
def get_png(view, filename):
    return serve_png_file(view, filename)

@app.route('/assets/<path:filename>')
def serve_asset(filename):
    # Serve static assets like Logo.png from the assets folder
    assets_dir = os.path.join(os.path.dirname(__file__), 'assets')
    return send_from_directory(assets_dir, filename)


@app.route("/run-task1")
def run_task1():
    def run_scripts_in_parallel():
        print("Flask is running as user:", getpass.getuser())  # Print user for debugging
        scripts = [
            ("/opt/render/project/src/NY/mslp_prate_csnow_EAST.py", "/opt/render/project/src/NY"),
            ("/opt/render/project/src/NY/tmp_2m_EAST.py", "/opt/render/project/src/NY"),
            ("/opt/render/project/src/NY/Snow_liquid_ratio_8to1_EAST.py", "/opt/render/project/src/NY"),
            ("/opt/render/project/src/NY/Snow_liquid_ratio_10to1_EAST.py", "/opt/render/project/src/NY"),
            ("/opt/render/project/src/NY/wind_10m_EAST.py", "/opt/render/project/src/NY"),
            ("/opt/render/project/src/NY/total_precip_EAST.py", "/opt/render/project/src/NY"),
            ("/opt/render/project/src/NY/vis_EAST.py", "/opt/render/project/src/NY"),
        ]

        # Semaphore to limit the number of concurrent processes
        semaphore = threading.Semaphore(3)  # Limit to 3 concurrent scripts

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
                            print(line, end="")  # Stream stdout to console
                        for line in process.stderr:
                            print(line, end="")  # Stream stderr to console
                        process.wait()
                        if process.returncode != 0:
                            print(f"Error: {os.path.basename(script)} exited with code {process.returncode}")
                        else:
                            print(f"{os.path.basename(script)} ran successfully!")
                except Exception as e:
                    print(f"Error running {os.path.basename(script)}: {e}")

        threads = []
        for script, cwd in scripts:
            thread = threading.Thread(target=run_script, args=(script, cwd))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()
            

    # Run the task in a separate thread
    threading.Thread(target=run_scripts_in_parallel).start()
    return "Task started in background! Check logs folder for output.", 200


@app.route("/run-task2")
def run_task2():
    def run_scripts_in_parallel():
        print("Flask is running as user:", getpass.getuser())  # Print user for debugging
        scripts = [
            ("/opt/render/project/src/GFS_USA/Prate_USA.py", "/opt/render/project/src/GFS_USA"),
            ("/opt/render/project/src/GFS_USA/tmp_2m_USA.py", "/opt/render/project/src/GFS_USA"),
            ("/opt/render/project/src/GFS_USA/absv_850mb_plot.py", "/opt/render/project/src/GFS_USA"),
                
            # Add more GFS scripts here as needed
        ]

        # Semaphore to limit the number of concurrent processes
        semaphore = threading.Semaphore(2)  # Limit to 2 concurrent scripts for GFS

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
                            print(line, end="")  # Stream stdout to console
                        for line in process.stderr:
                            print(line, end="")  # Stream stderr to console
                        process.wait()
                        if process.returncode != 0:
                            print(f"Error: {os.path.basename(script)} exited with code {process.returncode}")
                        else:
                            print(f"{os.path.basename(script)} ran successfully!")
                except Exception as e:
                    print(f"Error running {os.path.basename(script)}: {e}")

        threads = []
        for script, cwd in scripts:
            thread = threading.Thread(target=run_script, args=(script, cwd))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()
            

    # Run the task in a separate thread
    threading.Thread(target=run_scripts_in_parallel).start()
    return "Task 2 started in background! Check logs folder for output.", 200


if __name__ == "__main__":
    app.run(debug=True)
