#!/usr/bin/env python
"""Date-selectable web interface for the IDC403 multi-asset forecast dashboard.

Expected repository layout:
  IDC403_dashboard_updated.py
  updated_data/usd_inr_candidate_refreshed.csv
  updated_data/market_drives_candidate_refreshed.csv
  gold_refresh.csv
  app/date_forecast_webapp.py
"""
from __future__ import annotations

import html
import os
import re
import subprocess
import sys
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent if APP_DIR.name == "app" else APP_DIR
MODEL_SCRIPT = PROJECT_ROOT / "IDC403_dashboard_updated.py"
OUTPUT_ROOT = PROJECT_ROOT / "updated_data" / "date_picker_runs"
HOST = "0.0.0.0"
PORT = int(os.environ.get("PORT", "8765"))
SIMULATION_PATHS = 200

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>IDC403 | Multi-Asset Forecast Dashboard</title>
<style>
:root{color-scheme:light;--ink:#14243a;--muted:#62738a;--line:#dfe6ef;--blue:#1769e0}
*{box-sizing:border-box}
body{margin:0;background:#f2f5f9;color:var(--ink);font:16px/1.55 Inter,Arial,sans-serif}
main{max-width:900px;margin:clamp(16px,5vw,56px) auto;padding:clamp(20px,4vw,36px);background:#fff;border:1px solid var(--line);border-radius:18px;box-shadow:0 12px 36px rgba(20,36,58,.06)}
h1{font-size:clamp(26px,4vw,36px);line-height:1.15;margin:0 0 10px}
.muted{color:var(--muted)}
.eyebrow{font-size:12px;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:var(--blue);margin-bottom:10px}
.note{background:#fff4d6;padding:14px 16px;border-radius:10px;margin:22px 0;border:1px solid #f4e4b8}
.form{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px;align-items:end;margin:24px 0}
label{display:block;font-size:14px;font-weight:700;margin-bottom:7px}
input,select,button{width:100%;font:inherit;border-radius:9px;padding:12px 13px;border:1px solid #cbd5e1;background:#fff;color:var(--ink)}
button{grid-column:1/-1;background:var(--blue);color:#fff;border:0;cursor:pointer;font-weight:700}
button:disabled{opacity:.6;cursor:wait}
.small{font-size:13px}
.meta{display:flex;gap:10px;flex-wrap:wrap;margin-top:20px}
.pill{font-size:12px;padding:5px 10px;border:1px solid var(--line);border-radius:999px;color:var(--muted)}
@media(max-width:600px){.form{grid-template-columns:1fr}button{grid-column:auto}}
</style>
</head>
<body><main>
<div class="eyebrow">IDC403 · Quantitative Finance Research</div>
<h1>Multi-Asset Forecast Dashboard</h1>
<p class="muted">Generate a forecast from a selected historical origin using the project's USD/INR, gold and Brent crude modelling pipeline.</p>
<div class="note"><b>Historical-origin warning:</b> the model uses observations available on or before the selected date. A historical-origin forecast is not a forecast of today's market.</div>
<form class="form" method="post" action="/generate" onsubmit="const b=document.getElementById('go');b.disabled=true;b.textContent='Generating forecast…';document.getElementById('status').textContent='The model is running. Please keep this page open.'">
<div><label for="origin">Forecast origin date</label><input id="origin" name="origin" type="date" min="2020-01-01" max="__TODAY__" value="__LATEST__" required></div>
<div><label for="horizon">Forecast horizon</label><select id="horizon" name="horizon"><option value="30">30 business days</option><option value="90">90 business days</option><option value="252">1 year (252 days)</option><option value="504">2 years (504 days)</option><option value="756" selected>3 years (756 days)</option></select></div>
<button id="go" type="submit">Generate forecast</button>
</form>
<div id="status" class="muted" aria-live="polite"></div>
<div class="meta"><span class="pill">200 simulation paths</span><span class="pill">USD/INR</span><span class="pill">Gold · USD/troy oz</span><span class="pill">Brent · USD/barrel</span></div>
<p class="small muted">Forecast outputs are written to a run-specific folder. Simulation count is fixed at 200. Results are experimental and should not be treated as investment advice.</p>
</main></body></html>"""

def latest_fx_date() -> str:
    import pandas as pd
    csv_path = PROJECT_ROOT / "updated_data" / "usd_inr_candidate_refreshed.csv"
    if not csv_path.exists():
        return date.today().isoformat()
    df = pd.read_csv(csv_path, usecols=["date"])
    dates = pd.to_datetime(df["date"], errors="coerce").dropna()
    if dates.empty:
        return date.today().isoformat()
    return min(dates.max().date(), date.today()).isoformat()

def page_html() -> bytes:
    today = date.today().isoformat()
    try:
        latest = latest_fx_date()
    except Exception:
        latest = today
    return PAGE.replace("__TODAY__", today).replace("__LATEST__", latest).encode("utf-8")

class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print("[web]", fmt % args)

    def send_bytes(self, code: int, payload: bytes, content_type: str = "text/html; charset=utf-8"):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            return self.send_bytes(200, page_html())
        match = re.fullmatch(r"/dashboard/([A-Za-z0-9_-]+)", parsed.path)
        if match:
            target = OUTPUT_ROOT / match.group(1) / "multi_asset_dashboard.html"
            if not target.is_file():
                return self.send_bytes(404, b"Dashboard not found.")
            return self.send_bytes(200, target.read_bytes())
        return self.send_bytes(404, b"Not found.")

    def do_POST(self):
        if urlparse(self.path).path != "/generate":
            return self.send_bytes(404, b"Not found.")
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if size > 10_000:
                return self.send_bytes(413, b"Request too large.")
            form = parse_qs(self.rfile.read(size).decode("utf-8"))
            origin = form.get("origin", [""])[0]
            horizon = int(form.get("horizon", ["756"])[0])
            selected = date.fromisoformat(origin)
            if selected > date.today() or selected < date(2020, 1, 1):
                raise ValueError("Choose a valid date between 2020 and today.")
            if horizon not in (30, 90, 252, 504, 756):
                raise ValueError("Invalid forecast horizon.")
            if not MODEL_SCRIPT.is_file():
                raise FileNotFoundError(f"Missing model script: {MODEL_SCRIPT}")
            run_id = f"origin_{selected:%Y%m%d}_h{horizon}_s{SIMULATION_PATHS}"
            outdir = f"updated_data/date_picker_runs/{run_id}"
            cmd = [
                sys.executable, str(MODEL_SCRIPT), "--build-dashboard",
                "--origin-date", selected.isoformat(),
                "--horizon", str(horizon), "--outdir", outdir,
            ]
            print("[run]", " ".join(cmd))
            result = subprocess.run(
                cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=1800
            )
            if result.returncode != 0:
                msg = (result.stdout + "\n" + result.stderr)[-12000:]
                payload = (
                    "<!doctype html><meta charset='utf-8'><title>Forecast error</title>"
                    "<style>body{font:15px Arial;max-width:900px;margin:40px auto;padding:20px}"
                    "pre{white-space:pre-wrap;background:#f4f4f4;padding:16px}</style>"
                    "<h1>Forecast generation failed</h1><p><a href='/'>Back to date selection</a></p>"
                    "<pre>" + html.escape(msg) + "</pre>"
                ).encode("utf-8")
                return self.send_bytes(500, payload)
            target = OUTPUT_ROOT / run_id / "multi_asset_dashboard.html"
            if not target.is_file():
                msg = html.escape((result.stdout + "\n" + result.stderr)[-12000:])
                return self.send_bytes(500, f"<h1>Dashboard not found</h1><pre>{msg}</pre>".encode())
            self.send_response(303)
            self.send_header("Location", f"/dashboard/{run_id}")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
        except subprocess.TimeoutExpired:
            self.send_bytes(504, b"Forecast generation timed out after 30 minutes.")
        except Exception as exc:
            self.send_bytes(400, ("<h1>Request error</h1><p>" + html.escape(str(exc)) +
                                  "</p><p><a href='/'>Back</a></p>").encode("utf-8"))

def main():
    if not MODEL_SCRIPT.is_file():
        print(f"ERROR: Missing model script: {MODEL_SCRIPT}")
        raise SystemExit(2)
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Date-selectable forecasting app listening on {HOST}:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping web app...")
    finally:
        server.server_close()

if __name__ == "__main__":
    main()
