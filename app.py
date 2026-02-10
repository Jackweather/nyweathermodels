from flask import Flask, jsonify, send_from_directory, render_template
import os
import subprocess
import traceback
import threading
import getpass
import datetime

app = Flask(__name__)

# Directories containing the PNG files
BASE_DIR = "/var/data"
MSLP_PNG_DIR = os.path.join(BASE_DIR, "mslp_prate_csnow_NY", "png")
TEMP_PNG_DIR = os.path.join(BASE_DIR, "tmp_2m_NY", "png")
SNOW_8_TO_1_PNG_DIR = os.path.join(BASE_DIR, "snow_8_to_1_NY", "png")
SNOW_10_TO_1_PNG_DIR = os.path.join(BASE_DIR, "SNOW_10to_1_NY", "png")

@app.route("/")
def index():
    # Render the HTML frontend
    return render_template("index.html")

@app.route("/list_pngs/<view>")
def list_pngs(view):
    # List all PNG files in the directory based on the view (mslp or temp)
    if view == "temp":
        png_dir = TEMP_PNG_DIR
    else:
        png_dir = MSLP_PNG_DIR
    png_files = sorted([f"/get_png/{view}/{f}" for f in os.listdir(png_dir) if f.endswith(".png")])
    return jsonify(png_files)

@app.route("/get_png/<view>/<filename>")
def get_png(view, filename):
    # Serve a specific PNG file based on the view (mslp or temp)
    if view == "temp":
        png_dir = TEMP_PNG_DIR
    else:
        png_dir = MSLP_PNG_DIR
    return send_from_directory(png_dir, filename)

@app.route("/list_pngs/snow_8_to_1")
def list_pngs_snow_8_to_1():
    # List all PNG files in the snow_8_to_1 directory
    png_files = sorted([f"/get_png/snow_8_to_1/{f}" for f in os.listdir(SNOW_8_TO_1_PNG_DIR) if f.endswith(".png")])
    return jsonify(png_files)

@app.route("/get_png/snow_8_to_1/<filename>")
def get_png_snow_8_to_1(filename):
    # Serve a specific PNG file from the snow_8_to_1 directory
    return send_from_directory(SNOW_8_TO_1_PNG_DIR, filename)

@app.route("/list_pngs/snow_10_to_1")
def list_pngs_snow_10_to_1():
    # List all PNG files in the snow_10_to_1 directory
    png_files = sorted([f"/get_png/snow_10_to_1/{f}" for f in os.listdir(SNOW_10_TO_1_PNG_DIR) if f.endswith(".png")])
    return jsonify(png_files)

@app.route("/get_png/snow_10_to_1/<filename>")
def get_png_snow_10_to_1(filename):
    # Serve a specific PNG file from the snow_10_to_1 directory
    return send_from_directory(SNOW_10_TO_1_PNG_DIR, filename)

@app.route("/run-task1")
def run_task1():
    def run_all_scripts():
        print("Flask is running as user:", getpass.getuser())  # Print user for debugging
        scripts = [
            ("/opt/render/project/src/NY/mslp_prate_csnow_NY.py", "/opt/render/project/src/NY"),
            ("/opt/render/project/src/NY/tmp_2m_NY.py", "/opt/render/project/src/NY"),
            ("/opt/render/project/src/NY/snow_8_to_1_NY.py", "/opt/render/project/src/NY"),
            ("/opt/render/project/src/NY/SNOW_10to_1_NY.py", "/opt/render/project/src/NY"),
        ]
        for script, cwd in scripts:
            try:
                result = subprocess.run(
                    ["python", script],
                    check=True, cwd=cwd,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
                )
                print(f"{os.path.basename(script)} ran successfully!")
                print("STDOUT:", result.stdout)
                print("STDERR:", result.stderr)
            except subprocess.CalledProcessError as e:
                error_trace = traceback.format_exc()
                print(f"Error running {os.path.basename(script)}:\n{error_trace}")
                print("STDOUT:", e.stdout)
                print("STDERR:", e.stderr)

    # Run the task in a separate thread
    threading.Thread(target=run_all_scripts).start()
    return "Task started in background! Check logs folder for output.", 200

if __name__ == "__main__":
    app.run(debug=True)
