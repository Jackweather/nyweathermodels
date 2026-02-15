from flask import Flask, jsonify, send_from_directory, render_template
import os
import subprocess
import traceback
import threading
import getpass

app = Flask(__name__)

# Directories containing the PNG files
BASE_DIR = "/var/data"
PNG_DIRS = {
    "mslp": os.path.join(BASE_DIR, "mslp_prate_csnow_NY", "png"),
    "temp": os.path.join(BASE_DIR, "tmp_2m_NY", "png"),
    "snow_8_to_1": os.path.join(BASE_DIR, "snow_8_to_1_NY", "png"),
    "snow_10_to_1": os.path.join(BASE_DIR, "SNOW_10to_1_NY", "png"),
    "precip_type_rate": os.path.join(BASE_DIR, "mslp_prate_csnow_NY", "png"),
    "wind": os.path.join(BASE_DIR, "wind_10m_NY", "png"),
    "total_precip": os.path.join(BASE_DIR, "total_precip_NY", "png"),
}

@app.route("/")
def index():
    # Render the HTML frontend
    return render_template("index.html")

def list_png_files(view):
    # Helper function to list PNG files in a directory
    png_dir = PNG_DIRS.get(view)
    if not png_dir:
        return jsonify({"error": "Invalid view"}), 404
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

@app.route("/run-task1")
def run_task1():
    def run_scripts_in_parallel():
        print("Flask is running as user:", getpass.getuser())  # Print user for debugging
        scripts = [
            ("/opt/render/project/src/NY/mslp_prate_csnow_NY.py", "/opt/render/project/src/NY"),
            ("/opt/render/project/src/NY/tmp_2m_NY.py", "/opt/render/project/src/NY"),
            ("/opt/render/project/src/NY/Snow_liquid_ratio_8to1.py", "/opt/render/project/src/NY"),
            ("/opt/render/project/src/NY/Snow_liquid_ratio_10to1.py", "/opt/render/project/src/NY"),
            ("/opt/render/project/src/NY/wind_10m_NY.py", "/opt/render/project/src/NY"),
            ("/opt/render/project/src/NY/total_precip_NY.py", "/opt/render/project/src/NY"),
        ]

        # Semaphore to limit the number of concurrent processes
        semaphore = threading.Semaphore(6)  # Limit to 3 concurrent scripts

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


if __name__ == "__main__":
    app.run(debug=True)




