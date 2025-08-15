
import io
import re
import urllib.parse
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Route Links & GPX", page_icon="🗺️", layout="wide")
st.title("🗺️ Route Links & GPX — Validate & Fetch")

# --------------------------------------------------
# Utilities shared with your other page
# --------------------------------------------------
def cfg_value(df_cfg: pd.DataFrame, key: str, default=None):
    if df_cfg is None or df_cfg.empty:
        return default
    row = df_cfg.loc[df_cfg["Setting"] == key, "Value"]
    return row.values[0] if not row.empty else default

def clean(x):
    if pd.isna(x): return ""
    return str(x).strip()

def extract_sheet_id(url: str):
    m = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", url)
    return m.group(1) if m else None

def load_google_sheet_csv(sheet_id: str, sheet_name: str) -> pd.DataFrame:
    encoded = urllib.parse.quote(sheet_name, safe="")
    export_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={encoded}"
    df = pd.read_csv(export_url)
    if len(df.columns) and df.columns[0].lower().startswith("unnamed"):
        df = df.drop(columns=[df.columns[0]])
    return df

def load_from_google_csv(url: str):
    sheet_id = extract_sheet_id(url)
    required_tabs = ["Route Master"]
    dfs = {}
    for tab in required_tabs:
        df = load_google_sheet_csv(sheet_id, tab)
        dfs[tab] = df
    # Optional schedule/config if present
    for tab in ["Schedule", "Config"]:
        try:
            df = load_google_sheet_csv(sheet_id, tab)
            dfs[tab] = df
        except Exception:
            pass
    return dfs

def load_from_excel_bytes(bts: bytes):
    xls = pd.ExcelFile(io.BytesIO(bts))
    dfs = {"Route Master": pd.read_excel(xls, "Route Master") if "Route Master" in xls.sheet_names else pd.read_excel(xls, "RouteMaster")}
    for opt in ["Schedule", "Config"]:
        if opt in xls.sheet_names:
            dfs[opt] = pd.read_excel(xls, opt)
    return dfs

# --------------------------------------------------
# Link detection / patterns
# --------------------------------------------------
@dataclass
class LinkInfo:
    link_type: str
    url: str
    source_id: str
    is_possible_gpx: bool

STRAVA_ROUTE_RE = re.compile(r"strava\.com/routes/(\d+)", re.I)
STRAVA_ACTIVITY_RE = re.compile(r"strava\.com/activities/(\d+)", re.I)
PLOTAROUTE_RE = re.compile(r"plotaroute\.com/route/(\d+)", re.I)
DIRECT_GPX_RE = re.compile(r"\.gpx(\?.*)?$", re.I)

def parse_link(link_type: str, url: str) -> LinkInfo:
    lt = (link_type or "").strip().lower()
    url = (url or "").strip()
    sid = ""
    if lt in ["strava route", "strava"] or STRAVA_ROUTE_RE.search(url):
        m = STRAVA_ROUTE_RE.search(url)
        if m: sid = m.group(1)
        return LinkInfo("Strava Route", url, sid, False)
    if lt in ["strava activity"] or STRAVA_ACTIVITY_RE.search(url):
        m = STRAVA_ACTIVITY_RE.search(url)
        if m: sid = m.group(1)
        return LinkInfo("Strava Activity", url, sid, False)
    if lt in ["plotaroute"] or PLOTAROUTE_RE.search(url):
        m = PLOTAROUTE_RE.search(url)
        if m: sid = m.group(1)
        return LinkInfo("Plotaroute", url, sid, False)
    # Direct GPX
    if lt in ["gpx"] or DIRECT_GPX_RE.search(url):
        return LinkInfo("GPX", url, "", True)
    # Fallback
    return LinkInfo(link_type or "Unknown", url, sid, DIRECT_GPX_RE.search(url) is not None)

def head_status(url: str, timeout=12) -> Dict:
    try:
        r = requests.head(url, allow_redirects=True, timeout=timeout)
        status = r.status_code
        final_url = r.url
        ct = r.headers.get("Content-Type", "")
        # Some hosts block HEAD; try GET with stream to check headers
        if status >= 400 or status == 405 or not ct:
            r2 = requests.get(url, stream=True, allow_redirects=True, timeout=timeout)
            status = r2.status_code
            final_url = r2.url
            ct = r2.headers.get("Content-Type", "")
            r2.close()
        return {"ok": 200 <= status < 400, "status": status, "final_url": final_url, "content_type": ct}
    except Exception as e:
        return {"ok": False, "status": None, "final_url": "", "content_type": "", "error": str(e)}

def classify_access(host: str, info: Dict) -> str:
    if not info.get("ok"):
        return "Broken/Unreachable"
    final = (info.get("final_url") or "").lower()
    # Common login redirects
    if "strava.com/login" in final or "strava.com/session" in final:
        return "Needs Auth/Not Public"
    if "plotaroute.com" in host and info.get("status") in (401, 403):
        return "Not Public"
    return "OK"

# --------------------------------------------------
# UI: Load data
# --------------------------------------------------
mode = st.radio("Load Route Master from:", ["Google Sheet (CSV)", "Upload Excel (.xlsx)"], horizontal=True)

dfs = None
if mode == "Google Sheet (CSV)":
    gs_url = st.text_input("Google Sheet URL (must include a 'Route Master' tab)")
    if gs_url:
        try:
            dfs = load_from_google_csv(gs_url)
        except Exception as e:
            st.error(f"Could not read Google Sheet: {e}")
elif mode == "Upload Excel (.xlsx)":
    up = st.file_uploader("Upload your master file (.xlsx)", type=["xlsx"])
    if up is not None:
        try:
            dfs = load_from_excel_bytes(up.read())
        except Exception as e:
            st.error(f"Could not read Excel file: {e}")

if not dfs:
    st.stop()

rm = dfs.get("Route Master", pd.DataFrame())
if rm.empty:
    st.error("Route Master tab is empty or missing.")
    st.stop()

# Column names (tolerate variations)
COL_LINK_TYPE = next((c for c in rm.columns if "Route Link Type" in c), None)
COL_LINK = next((c for c in rm.columns if "Route Link" in c and "Type" not in c), None)
COL_NAME = next((c for c in rm.columns if c.strip().lower() == "route name"), None)

if not all([COL_LINK_TYPE, COL_LINK, COL_NAME]):
    st.error("Missing required columns in Route Master: 'Route Name', 'Route Link Type', 'Route Link (Source URL)'")
    st.stop()

st.caption("Tip: Works with Strava Route/Activity, Plotaroute, and direct GPX links. Strava GPX fetch requires OAuth (not enabled here) but we can still validate public visibility.")

# --------------------------------------------------
# Validate links
# --------------------------------------------------
st.subheader("Validate Links")
sample_limit = st.slider("Limit rows to validate (for speed)", min_value=10, max_value=rm.shape[0], value=min(200, rm.shape[0]), step=10)

subset = rm.head(sample_limit).copy()
rows = []
for _, r in subset.iterrows():
    name = clean(r[COL_NAME])
    ltype = clean(r[COL_LINK_TYPE])
    url = clean(r[COL_LINK])
    if not url:
        rows.append({"Route Name": name, "Link Type": ltype or "Unknown", "URL": "", "Status": "Missing URL", "HTTP": "", "Content-Type": "", "Source ID": ""})
        continue
    li = parse_link(ltype, url)
    info = head_status(li.url)
    status = classify_access(urllib.parse.urlparse(li.url).netloc, info)
    rows.append({
        "Route Name": name,
        "Link Type": li.link_type,
        "URL": li.url,
        "Status": status,
        "HTTP": info.get("status"),
        "Content-Type": info.get("content_type"),
        "Source ID": li.source_id
    })

report_df = pd.DataFrame(rows)
st.dataframe(report_df, use_container_width=True, hide_index=True)
st.download_button("⬇️ Download validation report (CSV)", data=report_df.to_csv(index=False).encode("utf-8"), file_name="route_link_validation.csv", mime="text/csv")

# --------------------------------------------------
# Basic GPX fetch (no OAuth)
# --------------------------------------------------
st.subheader("Fetch GPX (direct .gpx links only)")

gpx_candidates = report_df[(report_df["Status"] == "OK") & (report_df["URL"].str.lower().str.contains(".gpx"))]
if gpx_candidates.empty:
    st.caption("No direct .gpx links detected in the validated rows. You can still copy GPX from Strava/Plotaroute manually or enable OAuth later.")
else:
    pick = st.multiselect("Select routes to download GPX", gpx_candidates["Route Name"].tolist())
    if st.button("Download selected GPX"):
        zip_buf = io.BytesIO()
        import zipfile
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for _, row in gpx_candidates[gpx_candidates["Route Name"].isin(pick)].iterrows():
                url = row["URL"]
                try:
                    resp = requests.get(url, timeout=20)
                    resp.raise_for_status()
                    # Make a safe filename
                    parsed = urllib.parse.urlparse(url)
                    base = parsed.path.split("/")[-1] or "route.gpx"
                    fname = f"{row['Route Name'][:50].replace('/', '-')}-{base}"
                    zf.writestr(fname, resp.content)
                except Exception as e:
                    st.warning(f"Failed to download {row['Route Name']}: {e}")
        st.download_button("⬇️ Download GPX ZIP", data=zip_buf.getvalue(), file_name="routes_gpx.zip", mime="application/zip")

st.divider()
st.caption("For Strava Routes/Activities and Plotaroute, full automatic GPX export usually needs API OAuth and/or public routes. If you want, we can add Strava OAuth next so the app can fetch GPX for private routes you have access to.")
