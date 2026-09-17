# Body Fat % Estimator — Render-ready

## Files
- `app.py` — Flask app (trains the model from embedded data at startup, serves the UI)
- `templates/index.html`, `static/style.css`, `static/script.js` — the frontend
- `requirements.txt` — tells Render what to install
- `Procfile` — tells Render how to start the app

## Deploy steps
1. Put ALL of these files/folders at the **root** of your GitHub repo
   (not inside a subfolder — `app.py` and `requirements.txt` must be visible
   at the top level of the repo).
2. Push to GitHub.
3. In Render: New → Web Service → connect this repo.
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app`
4. Deploy. First boot takes ~10–20s longer than usual because the model
   trains fresh from the embedded dataset each time the service starts.

## Local test (optional)
```
pip install -r requirements.txt
python app.py
```
Then open http://127.0.0.1:5000
