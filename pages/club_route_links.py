
import io
import re
import urllib.parse
from dataclasses import dataclass
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Route Links & GPX", page_icon="🗺️", layout="wide")
st.title("🗺️ Route Links & GPX — Validate & Fetch (robust tabs/columns)")

# -----------------------------
# Helpers
# -----------------------------
def clean(x):
    if pd.isna(x): return ""
    return str(x).strip()

def norm_header(h):
    return re.sub(r"[^a-z0-9]+", "", str(h).strip().lower())

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
    dfs = {}
    for tab in ["Route Master", "RouteMaster", "Routemaster"]:
        try:
            df = load_google_sheet_csv(sheet_id, tab)
            if not df.empty:
                dfs["Route Master"] = df
                break
        except Exception:
            continue
    # Optional
    for tab in ["Schedule", "Config"]:
        try:
            df = load_google_sheet_csv(sheet_id, tab)
            if not df.empty:
                dfs[tab] = df
        except Exception:
            pass
    return dfs

def load_from_excel_bytes(bts: bytes):
    xls = pd.ExcelFile(io.BytesIO(bts))
    rm_name = None
    for name in xls.sheet_names:
        if name.lower().replace(" ", "") in {"routemaster", "route_master"}:
            rm_name = name; break
    if rm_name is None:
        rm_name = "Route Master" if "Route Master" in xls.sheet_names else "RouteMaster"
    dfs = {"Route Master": pd.read_excel(xls, rm_name)}
    for opt in ["Schedule", "Config"]:
        if opt in xls.sheet_names:
            dfs[opt] = pd.read_excel(xls, opt)
    return dfs

# -----------------------------
# Link parsing
# -----------------------------
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
    if lt in ["gpx"] or DIRECT_GPX_RE.search(url):
        return LinkInfo("GPX", url, "", True)
    return LinkInfo(link_type or "Unknown", url, sid, DIRECT_GPX_RE.search(url) is not None)

def head_status(url: str, timeout=12):
    try:
        r = requests.head(url, allow_redirects=True, timeout=timeout)
        status = r.status_code; final_url = r.url; ct = r.headers.get("Content-Type","")
        if status >= 400 or status == 405 or not ct:
            r2 = requests.get(url, stream=True, allow_redirects=True, timeout=timeout)
            status = r2.status_code; final_url = r2.url; ct = r2.headers.get("Content-Type",""); r2.close()
        return {"ok": 200 <= status < 400, "status": status, "final_url": final_url, "content_type": ct}
    except Exception as e:
        return {"ok": False, "status": None, "final_url": "", "content_type": "", "error": str(e)}

def classify_access(host: str, info: dict) -> str:
    if not info.get("ok"): return "Broken/Unreachable"
    final = (info.get("final_url") or "").lower()
    if "strava.com/login" in final or "strava.com/session" in final: return "Needs Auth/Not Public"
    if "plotaroute.com" in host and info.get("status") in (401,403): return "Not Public"
    return "OK"

# -----------------------------
# UI: load data
# -----------------------------
mode = st.radio("Load Route Master from:", ["Google Sheet (CSV)", "Upload Excel (.xlsx)"], horizontal=True)

dfs = None
if mode == "Google Sheet (CSV)":
    gs_url = st.text_input("Google Sheet URL")
    if gs_url:
        dfs = load_from_google_csv(gs_url)
elif mode == "Upload Excel (.xlsx)":
    up = st.file_uploader("Upload master Excel (.xlsx)", type=["xlsx"])
    if up is not None:
        dfs = load_from_excel_bytes(up.read())

if not dfs or "Route Master" not in dfs:
    st.error("Could not load a 'Route Master' tab. Accepted tab names: Route Master, RouteMaster, Routemaster.")
    st.stop()

rm = dfs["Route Master"]
if rm.empty:
    st.error("Route Master tab is empty."); st.stop()

# Flexible column detection
headers = list(rm.columns)
norm_map = {h: norm_header(h) for h in headers}

def find_col(targets):
    for h in headers:
        nh = norm_map[h]
        for t in targets:
            if t in nh:
                return h
    return None

COL_NAME = find_col(["routename"])
COL_LINK_TYPE = find_col(["routelinktype"])
# ensure URL col prefers source url over type
COL_LINK = find_col(["routelinksourceurl", "routelink", "url"])

if not COL_NAME or not COL_LINK_TYPE or not COL_LINK:
    st.error("Missing required columns. Detected headers: " + ", ".join(str(h) for h in headers))
    st.stop()

st.caption(f"Using columns → Name: '{COL_NAME}', Link Type: '{COL_LINK_TYPE}', URL: '{COL_LINK}'")

# Validate links
st.subheader("Validate Links")
sample_limit = st.slider("Limit rows to validate", min_value=10, max_value=rm.shape[0], value=min(200, rm.shape[0]), step=10)

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

# GPX download for direct links
st.subheader("Fetch GPX (direct .gpx links only)")
gpx_candidates = report_df[(report_df["Status"] == "OK") & (report_df["URL"].str.lower().str.contains(".gpx"))]
if gpx_candidates.empty:
    st.caption("No direct .gpx links detected in the validated rows. Public Strava/Plotaroute can be validated, GPX export needs OAuth unless you have direct URLs.")
else:
    pick = st.multiselect("Select routes to download GPX", gpx_candidates["Route Name"].tolist())
    if st.button("Download selected GPX"):
        zip_buf = io.BytesIO()
        import zipfile
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for _, row in gpx_candidates[gpx_candidates["Route Name"].isin(pick)].iterrows():
                url = row["URL"]
                try:
                    resp = requests.get(url, timeout=20); resp.raise_for_status()
                    base = urllib.parse.urlparse(url).path.split("/")[-1] or "route.gpx"
                    fname = f"{row['Route Name'][:50].replace('/', '-')}-{base}"
                    zf.writestr(fname, resp.content)
                except Exception as e:
                    st.warning(f"Failed to download {row['Route Name']}: {e}")
        st.download_button("⬇️ Download GPX ZIP", data=zip_buf.getvalue(), file_name="routes_gpx.zip", mime="application/zip")
