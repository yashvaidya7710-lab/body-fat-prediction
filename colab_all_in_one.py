"""
=====================================================================
 PASTE THIS ENTIRE SCRIPT INTO ONE GOOGLE COLAB CELL AND RUN IT.

 No signup, no token, no external service — this uses Colab's own
 built-in port proxy (google.colab.output) to expose the Flask app,
 which only works inside Colab (that's expected; it's a Colab feature,
 not a generic Python one).

 It will:
   1. Install Flask and the ML libraries
   2. Retrain the Random Forest model from embedded training data
      (no file upload needed at all — the ~66 KB dataset is baked
      into this script, compressed)
   3. Write app.py + templates/index.html + static/style.css +
      static/script.js to disk
   4. Start the Flask app in the background
   5. Print a clickable link that opens the app in a new tab,
      proxied through Colab — nothing else to configure
=====================================================================
"""

# ---------------------------------------------------------------
# STEP 1: INSTALL DEPENDENCIES
# ---------------------------------------------------------------
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "flask", "pandas", "scikit-learn", "joblib", "numpy"])

import os

# ---------------------------------------------------------------
# STEP 2: TRAIN THE MODEL — FROM EMBEDDED DATA, NO UPLOAD NEEDED
#    The cleaned/preprocessed training data (973 rows) is embedded
#    below as a compressed, base64-encoded blob. This cell decodes
#    it, retrains the exact same Random Forest Regressor used
#    earlier (same hyperparameters, same random_state=42), and gets
#    an identical model — no file upload required at all.
# ---------------------------------------------------------------
import base64
import gzip
import io
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

_EMBEDDED_DATA_B64 = (
    "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"
)

_csv_bytes = gzip.decompress(base64.b64decode(_EMBEDDED_DATA_B64))
df = pd.read_csv(io.BytesIO(_csv_bytes))
print(f"Loaded embedded training data: {df.shape[0]} rows, {df.shape[1]} columns")

TARGET = "Fat_Percentage"
X = df.drop(columns=[TARGET])
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

rf_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print(f"Model retrained. Test MAE={mae:.2f}  RMSE={rmse:.2f}  R2={r2:.2f}")

artifact = {
    "model": rf_model,
    "feature_names": list(X.columns),
    "target_name": TARGET,
    "metrics": {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2},
}

MODEL_FILENAME = "random_forest_fat_percentage_model.pkl"
import joblib
joblib.dump(artifact, MODEL_FILENAME)
print(f"Saved model to: {MODEL_FILENAME}")

# ---------------------------------------------------------------
# STEP 3: WRITE THE APP FILES TO DISK
# ---------------------------------------------------------------
os.makedirs("templates", exist_ok=True)
os.makedirs("static", exist_ok=True)

APP_PY = '''
from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "random_forest_fat_percentage_model.pkl")
artifact = joblib.load(MODEL_PATH)
model = artifact["model"]
FEATURE_NAMES = artifact["feature_names"]
MODEL_METRICS = artifact.get("metrics", {})

WORKOUT_TYPES = ["Cardio", "HIIT", "Strength", "Yoga"]


@app.route("/")
def index():
    return render_template("index.html", workout_types=WORKOUT_TYPES, metrics=MODEL_METRICS)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        age = float(data["age"])
        gender_label = data["gender"]
        weight_kg = float(data["weight_kg"])
        height_m = float(data["height_m"])
        max_bpm = float(data["max_bpm"])
        avg_bpm = float(data["avg_bpm"])
        resting_bpm = float(data["resting_bpm"])
        session_duration = float(data["session_duration"])
        calories_burned = float(data["calories_burned"])
        water_intake = float(data["water_intake"])
        workout_freq = float(data["workout_freq"])
        experience_level = float(data["experience_level"])
        workout_type = data["workout_type"]

        if height_m <= 0:
            return jsonify({"error": "Height must be greater than zero."}), 400
        if workout_type not in WORKOUT_TYPES:
            return jsonify({"error": f"Workout type must be one of {WORKOUT_TYPES}."}), 400

        bmi = weight_kg / (height_m ** 2)
        bpm_reserve = max_bpm - resting_bpm
        gender = 1 if gender_label == "Male" else 0

        row = {
            "Age": age, "Gender": gender, "Weight_kg": weight_kg, "Height_m": height_m,
            "Max_BPM": max_bpm, "Avg_BPM": avg_bpm, "Resting_BPM": resting_bpm,
            "Session_Duration_hrs": session_duration, "Calories_Burned": calories_burned,
            "Water_Intake_liters": water_intake, "Workout_Frequency_per_week": workout_freq,
            "Experience_Level": experience_level, "BMI": bmi,
            "Workout_Cardio": 1 if workout_type == "Cardio" else 0,
            "Workout_HIIT": 1 if workout_type == "HIIT" else 0,
            "Workout_Strength": 1 if workout_type == "Strength" else 0,
            "Workout_Yoga": 1 if workout_type == "Yoga" else 0,
            "BPM_Reserve": bpm_reserve,
        }

        X = pd.DataFrame([[row[col] for col in FEATURE_NAMES]], columns=FEATURE_NAMES)
        prediction = float(model.predict(X)[0])

        return jsonify({"prediction": round(prediction, 2), "bmi": round(bmi, 1), "bpm_reserve": round(bpm_reserve, 1)})

    except KeyError as e:
        return jsonify({"error": f"Missing field: {e}"}), 400
    except ValueError:
        return jsonify({"error": "All numeric fields must be valid numbers."}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500
'''

INDEX_HTML = '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Body Fat % Estimator</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">
</head>
<body>

<header class="site-header">
  <div class="wordmark">Body Fat Estimator</div>
  <p class="tagline">Predicts body fat percentage from training and physiology data, using a Random Forest model trained on 973 gym members.</p>
</header>

<main class="layout">

  <section class="log-sheet" aria-label="Input form">
    <form id="predict-form">

      <div class="log-section">
        <h2>Profile</h2>
        <div class="field-row">
          <div class="field"><label for="age">Age (years)</label>
            <input type="number" id="age" name="age" min="15" max="90" step="1" value="30" required></div>
          <div class="field"><label for="gender">Gender</label>
            <select id="gender" name="gender" required><option value="Male">Male</option><option value="Female">Female</option></select></div>
        </div>
        <div class="field-row">
          <div class="field"><label for="weight_kg">Weight (kg)</label>
            <input type="number" id="weight_kg" name="weight_kg" min="30" max="200" step="0.1" value="75" required></div>
          <div class="field"><label for="height_m">Height (m)</label>
            <input type="number" id="height_m" name="height_m" min="1.2" max="2.3" step="0.01" value="1.75" required></div>
        </div>
      </div>

      <div class="log-section">
        <h2>Heart rate</h2>
        <div class="field-row">
          <div class="field"><label for="max_bpm">Max BPM</label>
            <input type="number" id="max_bpm" name="max_bpm" min="120" max="220" step="1" value="180" required></div>
          <div class="field"><label for="avg_bpm">Average BPM (session)</label>
            <input type="number" id="avg_bpm" name="avg_bpm" min="80" max="200" step="1" value="145" required></div>
          <div class="field"><label for="resting_bpm">Resting BPM</label>
            <input type="number" id="resting_bpm" name="resting_bpm" min="35" max="100" step="1" value="62" required></div>
        </div>
      </div>

      <div class="log-section">
        <h2>Session</h2>
        <div class="field-row">
          <div class="field"><label for="session_duration">Session duration (hours)</label>
            <input type="number" id="session_duration" name="session_duration" min="0.1" max="4" step="0.05" value="1.2" required></div>
          <div class="field"><label for="calories_burned">Calories burned</label>
            <input type="number" id="calories_burned" name="calories_burned" min="50" max="2500" step="1" value="900" required></div>
          <div class="field"><label for="workout_type">Workout type</label>
            <select id="workout_type" name="workout_type" required>
              {% for w in workout_types %}<option value="{{ w }}">{{ w }}</option>{% endfor %}
            </select></div>
        </div>
      </div>

      <div class="log-section">
        <h2>Habits</h2>
        <div class="field-row">
          <div class="field"><label for="water_intake">Water intake (liters/day)</label>
            <input type="number" id="water_intake" name="water_intake" min="0.5" max="6" step="0.1" value="2.6" required></div>
          <div class="field"><label for="workout_freq">Workout frequency (days/week)</label>
            <input type="number" id="workout_freq" name="workout_freq" min="1" max="7" step="1" value="4" required></div>
          <div class="field"><label for="experience_level">Experience level (1&ndash;3)</label>
            <select id="experience_level" name="experience_level" required>
              <option value="1">1 — Beginner</option><option value="2" selected>2 — Intermediate</option><option value="3">3 — Advanced</option>
            </select></div>
        </div>
      </div>

      <button type="submit" id="submit-btn">Estimate body fat %</button>
      <p class="form-note" id="form-error" role="alert"></p>
    </form>
  </section>

  <aside class="result-panel" aria-live="polite">
    <div class="result-label">Estimated body fat</div>
    <div class="result-readout" id="result-readout">—</div>
    <div class="result-unit">percent</div>
    <div class="result-derived" id="result-derived" hidden>
      <div><span>BMI</span><strong id="derived-bmi">—</strong></div>
      <div><span>Heart-rate reserve</span><strong id="derived-hrr">—</strong></div>
    </div>
    <p class="result-hint" id="result-hint">Fill in the form and press Estimate body fat % to see a prediction.</p>
    {% if metrics %}
    <div class="model-note">Model: Random Forest Regressor &middot; test R&sup2; {{ "%.2f"|format(metrics.get('R2', 0)) }} &middot; RMSE {{ "%.2f"|format(metrics.get('RMSE', 0)) }}</div>
    {% endif %}
  </aside>

</main>

<script src="{{ url_for('static', filename='script.js') }}"></script>
</body>
</html>
'''

STYLE_CSS = ''':root {
  --bg: #12181F; --panel: #1B232C; --panel-raised: #212B35; --line: #2B3540;
  --text: #E7ECEF; --text-muted: #8FA0AC; --accent: #F2B134; --accent-2: #3FA796;
  --danger: #E0654F; --radius: 3px;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text); font-family: 'Inter', sans-serif; font-size: 16px; line-height: 1.5; }
.site-header { max-width: 920px; margin: 0 auto; padding: 3rem 1.5rem 1.5rem; }
.wordmark { font-family: 'Barlow Condensed', sans-serif; font-weight: 700; font-size: 2.4rem; letter-spacing: 0.01em; }
.tagline { max-width: 46ch; color: var(--text-muted); margin-top: 0.4rem; font-size: 0.98rem; }
.layout { max-width: 920px; margin: 0 auto; padding: 0 1.5rem 4rem; display: grid; grid-template-columns: 1.5fr 1fr; gap: 1.75rem; align-items: start; }
@media (max-width: 760px) { .layout { grid-template-columns: 1fr; } }
.log-sheet { background: var(--panel); border: 1px solid var(--line); border-radius: var(--radius); padding: 0.5rem 1.75rem 1.75rem; }
.log-section { padding: 1.25rem 0; border-bottom: 1px solid var(--line); }
.log-section:last-of-type { border-bottom: none; }
.log-section h2 { font-family: 'Barlow Condensed', sans-serif; font-weight: 600; font-size: 1.15rem; margin: 0 0 0.9rem; color: var(--accent-2); }
.field-row { display: flex; gap: 1rem; flex-wrap: wrap; }
.field { flex: 1 1 140px; display: flex; flex-direction: column; gap: 0.35rem; }
.field label { font-size: 0.85rem; color: var(--text-muted); }
.field input, .field select { background: var(--panel-raised); border: 1px solid var(--line); border-radius: var(--radius); color: var(--text); padding: 0.55rem 0.6rem; font-family: 'Inter', sans-serif; font-size: 0.95rem; }
.field input:focus, .field select:focus { outline: 2px solid var(--accent); outline-offset: 1px; border-color: var(--accent); }
#submit-btn { margin-top: 1.5rem; width: 100%; background: var(--accent); color: #1A1300; border: none; border-radius: var(--radius); padding: 0.85rem 1rem; font-family: 'Barlow Condensed', sans-serif; font-weight: 600; font-size: 1.1rem; letter-spacing: 0.01em; cursor: pointer; transition: background 0.15s ease; }
#submit-btn:hover { background: #FFC456; }
#submit-btn:disabled { background: var(--line); color: var(--text-muted); cursor: wait; }
.form-note { min-height: 1.2rem; color: var(--danger); font-size: 0.85rem; margin: 0.6rem 0 0; }
.result-panel { background: var(--panel); border: 1px solid var(--line); border-radius: var(--radius); padding: 1.75rem; text-align: center; position: sticky; top: 2rem; }
.result-label { font-size: 0.85rem; color: var(--text-muted); }
.result-readout { font-family: 'Barlow Condensed', sans-serif; font-weight: 700; font-size: 4.5rem; line-height: 1; font-variant-numeric: tabular-nums; color: var(--accent); margin: 0.4rem 0 0.1rem; transition: opacity 0.2s ease; }
.result-readout.updated { animation: pulse 0.4s ease; }
@keyframes pulse { 0% { opacity: 0.3; transform: scale(0.97); } 100% { opacity: 1; transform: scale(1); } }
.result-unit { color: var(--text-muted); font-size: 0.85rem; margin-bottom: 1.25rem; }
.result-derived { border-top: 1px solid var(--line); padding-top: 1rem; margin-top: 0.5rem; display: flex; flex-direction: column; gap: 0.5rem; text-align: left; }
.result-derived div { display: flex; justify-content: space-between; font-size: 0.9rem; }
.result-derived span { color: var(--text-muted); }
.result-derived strong { font-variant-numeric: tabular-nums; }
.result-hint { color: var(--text-muted); font-size: 0.85rem; margin-top: 1rem; }
.model-note { margin-top: 1.5rem; padding-top: 1rem; border-top: 1px solid var(--line); font-size: 0.78rem; color: var(--text-muted); }
'''

SCRIPT_JS = '''document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("predict-form");
  const submitBtn = document.getElementById("submit-btn");
  const errorNote = document.getElementById("form-error");
  const readout = document.getElementById("result-readout");
  const hint = document.getElementById("result-hint");
  const derivedBox = document.getElementById("result-derived");
  const derivedBmi = document.getElementById("derived-bmi");
  const derivedHrr = document.getElementById("derived-hrr");

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    errorNote.textContent = "";
    submitBtn.disabled = true;
    submitBtn.textContent = "Estimating…";

    const payload = {
      age: document.getElementById("age").value,
      gender: document.getElementById("gender").value,
      weight_kg: document.getElementById("weight_kg").value,
      height_m: document.getElementById("height_m").value,
      max_bpm: document.getElementById("max_bpm").value,
      avg_bpm: document.getElementById("avg_bpm").value,
      resting_bpm: document.getElementById("resting_bpm").value,
      session_duration: document.getElementById("session_duration").value,
      calories_burned: document.getElementById("calories_burned").value,
      workout_type: document.getElementById("workout_type").value,
      water_intake: document.getElementById("water_intake").value,
      workout_freq: document.getElementById("workout_freq").value,
      experience_level: document.getElementById("experience_level").value,
    };

    try {
      const response = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await response.json();

      if (!response.ok) {
        errorNote.textContent = data.error || "Something went wrong. Check the values and try again.";
        return;
      }

      readout.textContent = data.prediction.toFixed(1);
      readout.classList.remove("updated");
      void readout.offsetWidth;
      readout.classList.add("updated");

      derivedBmi.textContent = data.bmi;
      derivedHrr.textContent = data.bpm_reserve;
      derivedBox.hidden = false;
      hint.textContent = "Based on the values entered — not a medical measurement.";
    } catch (err) {
      errorNote.textContent = "Could not reach the server. Please try again.";
    } finally {
      submitBtn.disabled = false;
      submitBtn.textContent = "Estimate body fat %";
    }
  });
});
'''

with open("app.py", "w") as f:
    f.write(APP_PY)
with open("templates/index.html", "w") as f:
    f.write(INDEX_HTML)
with open("static/style.css", "w") as f:
    f.write(STYLE_CSS)
with open("static/script.js", "w") as f:
    f.write(SCRIPT_JS)

print("App files written: app.py, templates/index.html, static/style.css, static/script.js")

# ---------------------------------------------------------------
# STEP 4: RUN THE FLASK APP IN THE BACKGROUND
#    (must run in a background thread so this cell can move on to
#    print the link instead of blocking on app.run() forever)
# ---------------------------------------------------------------
import importlib
import sys as _sys
if "app" in _sys.modules:
    importlib.reload(_sys.modules["app"])
from app import app as flask_app

import threading
import time

PORT = 5000

def _run_app():
    flask_app.run(port=PORT, use_reloader=False)

thread = threading.Thread(target=_run_app, daemon=True)
thread.start()
time.sleep(2)  # give the server a moment to start before we open the proxy

# ---------------------------------------------------------------
# STEP 5: OPEN IT THROUGH COLAB'S BUILT-IN PORT PROXY
#    This only works inside a real Colab runtime — that's expected,
#    it's a Colab-provided feature, not a plain Python one.
# ---------------------------------------------------------------
try:
    from google.colab.output import serve_kernel_port_as_window
    print("Click the link below to open the app in a new tab:\n")
    serve_kernel_port_as_window(PORT)
except ImportError:
    # Not running in Colab (e.g. testing locally) — just tell them the local URL
    print(f"Not running in Colab. Open this URL in your browser: http://127.0.0.1:{PORT}")

# Keep this cell "running" so the background server thread stays alive.
# This is expected — leave the cell in this state while you use the app,
# and interrupt/stop the cell (or restart the runtime) when you're done.
print("\nThis cell will keep running to keep the server alive. Stop it when you're done.")
while True:
    time.sleep(60)
