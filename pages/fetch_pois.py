
# pages/fetch_pois.py
# Build: v2025.08.16-POI-GSHEET-ONROUTE
import io
import re
import time
import urllib.parse
import pandas as pd
import requests
import streamlit as st
from math import radians, sin, cos, asin, sqrt

st.set_page_config(page_title="📍 Route POIs — LocationIQ (On-route)", page_icon="📍", layout="wide")
st.title("📍 Route POIs — LocationIQ (On-route)")
st.caption("Build: v2025.08.16-POI-GSHEET-ONROUTE — Roads & landmarks strictly from points sampled *on* the route polyline.")

# ----------------------------- Helpers ----------------------------------
def clean(x):
    return "" if pd.isna(x) else str(x).strip()

def norm_header(h):
    return re.sub(r"[^a-z0-9]+", "", str(h).strip().lower())

def extract_sheet_id(url):
    m = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", url)
    return m.group(1) if m else None

def load_google_sheet_csv(sheet_id, sheet_name):
    u = (
        f"https://docs.google.com/spreadsheets/d/{sheet_id}"
        "/gviz/tq?tqx=out:csv&sheet=" + urllib.parse.quote(sheet_name, safe="")
    )
    df = pd.read_csv(u, dtype=str, keep_default_na=False)
    return df

def load_from_google_csv(url):
    sid = extract_sheet_id(url)
    dfs = {}
    if sid:
        df = load_google_sheet_csv(sid, "Schedule")
        if not df.empty:
            dfs["Schedule"] = df
    return dfs

def make_https(u):
    u = (u or "").strip()
    if not u:
        return u
    if " " in u or u.lower().startswith("strava route"):
        return ""  # placeholder/invalid
    if not urllib.parse.urlparse(u).scheme:
        return "https://" + u
    return u

STRAVA_ROUTE_ID_RE = re.compile(r"(?:^|/)(?:routes|routes/view)/(\d+)(?:[/?#].*)?$", re.I)

def is_strava_route_url(u: str) -> bool:
    if not u:
        return False
    lu = u.lower()
    return ("strava.com" in lu) and ("/routes/" in lu)

def extract_route_id_from_url(u: str) -> str:
    m = STRAVA_ROUTE_ID_RE.search(u or "")
    return m.group(1) if m else ""

def extract_digits(s: str) -> str:
    return "".join(re.findall(r"\d+", s or ""))

def expand_sci_id(s: str) -> str:
    if not s:
        return ""
    s = s.strip()
    m = re.match(r"^(\d+)(?:\.(\d+))?[eE]\+?(\d+)$", s)
    if not m:
        return ""
    int_part, frac_part, exp_str = m.group(1), (m.group(2) or ""), m.group(3)
    try:
        exp = int(exp_str)
    except ValueError:
        return ""
    digits = int_part + frac_part
    zeros_to_add = exp - len(frac_part)
    if zeros_to_add >= 0:
        return digits + ("0" * zeros_to_add)
    else:
        return digits

def source_id_to_digits(s: str) -> str:
    expanded = expand_sci_id(s)
    return expanded if expanded else extract_digits(s)

def extract_route_id(url: str, source_id: str) -> str:
    rid = extract_route_id_from_url(url)
    if rid:
        return rid
    return source_id_to_digits(source_id)

# Pure-Python polyline decoder
def decode_polyline(polyline_str):
    coords = []
    index = 0
    lat = 0
    lng = 0
    length = len(polyline_str)

    while index < length:
        result = 0
        shift = 0
        while True:
            b = ord(polyline_str[index]) - 63
            index += 1
            result |= (b & 0x1f) << shift
            shift += 5
            if b < 0x20:
                break
        dlat = ~(result >> 1) if (result & 1) else (result >> 1)
        lat += dlat

        result = 0
        shift = 0
        while True:
            b = ord(polyline_str[index]) - 63
            index += 1
            result |= (b & 0x1f) << shift
            shift += 5
            if b < 0x20:
                break
        dlng = ~(result >> 1) if (result & 1) else (result >> 1)
        lng += dlng

        coords.append((lat / 1e5, lng / 1e5))
    return coords

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    dlat = radians(lat2-lat1)
    dlon = radians(lon2-lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return R * c

def sample_polyline(coords, step_m=600, max_points=36):
    if not coords:
        return []
    samples = [coords[0]]
    acc = 0.0
    for i in range(1, len(coords)):
        d = haversine_m(coords[i-1][0], coords[i-1][1], coords[i][0], coords[i][1])
        acc += d
        if acc >= step_m:
            samples.append(coords[i])
            acc = 0.0
        if len(samples) >= max_points:
            break
    if samples[-1] != coords[-1] and len(samples) < max_points:
        samples.append(coords[-1])
    return samples

# API helpers
def get_liq_key():
    try:
        return st.secrets["locationiq"]["api_key"]
    except Exception:
        return None

def strava_get_route_json(route_id: str, access_token: str):
    url = f"https://www.strava.com/api/v3/routes/{route_id}"
    return requests.get(url, headers={"Authorization": f"Bearer {access_token}"}, timeout=25)

def locationiq_reverse(lat: float, lon: float, api_key: str):
    u = "https://us1.locationiq.com/v1/reverse"
    params = {
        "key": api_key,
        "lat": f"{lat:.6f}",
        "lon": f"{lon:.6f}",
        "format": "json",
        "normalizeaddress": 1,
        "zoom": 18,  # tighter to road level
    }
    return requests.get(u, params=params, timeout=25)

# ---------------------------- Load from Google Sheet ----------------------
st.subheader("Load your Schedule (Google Sheet)")
gs_url = st.text_input("Google Sheet URL")
dfs = None
if gs_url:
    try:
        dfs = load_from_google_csv(gs_url)
    except Exception as e:
        st.error(f"Could not read Google Sheet: {e}")

if not dfs or "Schedule" not in dfs:
    st.error("Could not load a 'Schedule' tab. Enter your Google Sheet URL above.")
    st.stop()

sched = dfs["Schedule"]
sched.columns = [str(c) for c in sched.columns]
cols = {c: norm_header(c) for c in sched.columns}

def find_col(targets):
    for c, n in cols.items():
        for t in targets:
            if t in n:
                return c
    return None

date_col = find_col(["date","datethu"])
r_names = [find_col(["route1name"]), find_col(["route2name"])]
r_types = [find_col(["route1routelinktype","route1linktype"]), find_col(["route2routelinktype","route2linktype"])]
r_urls  = [find_col(["route1routelinksourceurl","route1routelink","route1url"]), find_col(["route2routelinksourceurl","route2routelink","route2url"])]
r_srcid = [find_col(["route1sourceid","route1id"]), find_col(["route2sourceid","route2id"])]

if not date_col or not all(r_names) or not all(r_types) or not all(r_urls):
    st.error("Schedule is missing expected columns for route names/link types/URLs.")
    st.stop()

# Flatten rows & synthesize Strava URLs if missing
synthesized = 0
rows = []
for _, row in sched.iterrows():
    d = row.get(date_col, "")
    for i, side in enumerate(["1","2"]):
        nm = clean(row.get(r_names[i], ""))
        lt = clean(row.get(r_types[i], ""))
        url_raw = clean(row.get(r_urls[i], ""))
        sid_raw = clean(row.get(r_srcid[i], "")) if r_srcid[i] else ""
        if not nm or nm.lower() == "no run":
            continue
        url = make_https(url_raw)
        if (not url) and ("strava" in (lt or "").lower() or "strava" in (nm or "").lower()):
            sid_digits = source_id_to_digits(sid_raw)
            if len(sid_digits) >= 6:
                url = f"https://www.strava.com/routes/{sid_digits}"
                synthesized += 1
        rid = extract_route_id(url, sid_raw) if ("strava" in (lt or "").lower() or is_strava_route_url(url)) else ""
        rows.append({"Date": d, "Route Name": nm, "Side": side, "Link Type": lt, "URL": url, "Route ID": rid})

df_links = pd.DataFrame(rows)
st.write(f"Found {len(df_links)} route entries. (Synthesized {synthesized} Strava URLs)")

# --------------------------- Controls ------------------------------------
st.subheader("POI Fetch Controls")
access_token = st.session_state.get("strava_token")
if not access_token:
    st.info("Connect your Strava account on the OAuth page first.")
liq_key = get_liq_key()
if not liq_key:
    st.warning("Missing LocationIQ key. Add to Streamlit secrets as:\n\n[locationiq]\napi_key = \"YOUR_KEY\"")

start_at = st.number_input("Start at row", min_value=0, max_value=max(0, len(df_links)-1), value=0, step=1)
limit = st.slider("Rows in this pass", min_value=10, max_value=len(df_links), value=min(40, len(df_links)), step=10)
step_m = st.slider("Sampling interval (meters)", min_value=200, max_value=1200, value=600, step=50)
max_pts = st.slider("Max sampled points per route", min_value=12, max_value=60, value=36, step=4)
delay_route = st.slider("Delay between routes (seconds)", min_value=0.0, max_value=2.0, value=0.35, step=0.05)
delay_liq = st.slider("Delay between LocationIQ calls (seconds)", min_value=0.0, max_value=1.0, value=0.7, step=0.05)

subset = df_links.iloc[start_at:start_at+limit].copy()

# Allowed landmark classes/types (OSM/LocationIQ schema)
ALLOWED_CLASSES = {
    "leisure": {"park", "common", "nature_reserve"},
    "natural": {"wood", "heath", "grassland", "scrub"},
    "waterway": {"river", "canal"},
    "landuse": {"forest"},
    "water": {"reservoir", "lake"},
    "amenity": {"stadium", "sports_centre"},
    "tourism": {"attraction"},
}

def is_allowed_landmark(p):
    cls = p.get("class")
    typ = p.get("type")
    if not cls or not typ:
        return False
    return cls in ALLOWED_CLASSES and typ in ALLOWED_CLASSES[cls]

def extract_road_names(p):
    a = p.get("address", {}) or {}
    candidates = []
    for key in ("road", "pedestrian", "footway", "path", "cycleway"):
        v = a.get(key)
        if v:
            candidates.append(v)
    return candidates

def poi_label(p):
    name = p.get("name") or p.get("display_name") or ""
    typ = p.get("type") or p.get("category") or p.get("class")
    if name and typ:
        return f"{name} ({typ})"
    return name or ""

# --------------------------- Fetch POIs ----------------------------------
st.subheader("Fetch POIs (on-route only)")
out_rows = []
prog = st.progress(0, text="Ready…")
total = len(subset)

tot_calls = 0
ok_calls = 0
throttled = 0
errs = 0

for i, (_, r) in enumerate(subset.iterrows(), start=1):
    url = r["URL"]; rid = r["Route ID"]
    status = "Skipped"
    roads = set()
    landmarks = set()
    samples_used = 0

    if url and is_strava_route_url(url) and rid and access_token and liq_key:
        # Get route JSON
        resp = strava_get_route_json(rid, access_token)
        if resp.status_code == 200:
            data = resp.json()
            poly = (data.get("map") or {}).get("polyline") or (data.get("map") or {}).get("summary_polyline")
            if poly:
                try:
                    coords = decode_polyline(poly)
                except Exception:
                    coords = []
                pts = sample_polyline(coords, step_m=step_m, max_points=max_pts)
                samples_used = len(pts)
                for (lat, lon) in pts:
                    try:
                        rr = locationiq_reverse(lat, lon, liq_key)
                        tot_calls += 1
                        if rr.status_code == 200:
                            ok_calls += 1
                            p = rr.json()
                            # On-route extraction: roads from the exact sample point only
                            for rn in extract_road_names(p):
                                roads.add(rn)
                            # Landmarks only if class/type allowed
                            if is_allowed_landmark(p):
                                lbl = poi_label(p)
                                if lbl:
                                    landmarks.add(lbl)
                        elif rr.status_code in (402, 429):  # quota/rate
                            throttled += 1
                        else:
                            errs += 1
                        if delay_liq > 0:
                            time.sleep(delay_liq)
                    except Exception:
                        errs += 1
                status = "OK" if (roads or landmarks) else "No POIs found"
            else:
                status = "No polyline in Strava route"
        elif resp.status_code == 404:
            status = "Not Found/No Access (Strava)"
        elif resp.status_code == 401:
            status = "Unauthorized (Strava token)"
        elif resp.status_code == 429:
            status = "Rate-limited (Strava)"
        else:
            status = f"Strava API error {resp.status_code}"
    else:
        if not url:
            status = "Missing URL"
        elif not rid:
            status = "Missing Route ID"
        elif not access_token:
            status = "Needs Strava login"
        elif not liq_key:
            status = "Missing LocationIQ key"
        else:
            status = "Skipped"

    out_rows.append({
        "Date": r["Date"],
        "Route Name": r["Route Name"],
        "URL": url,
        "Route ID": rid,
        "Status": status,
        "Roads (on-route)": "; ".join(sorted(roads)) if roads else None,
        "Landmarks (on-route)": "; ".join(sorted(landmarks)) if landmarks else None,
        "Samples Used": samples_used,
    })

    prog.progress(min(i/total, 1.0), text=f"{i}/{total} processed")
    if delay_route > 0:
        time.sleep(delay_route)

prog.empty()

res = pd.DataFrame(out_rows)
st.dataframe(res, use_container_width=True, hide_index=True)

st.download_button(
    "⬇️ Download Roads & Landmarks (CSV)",
    data=res.to_csv(index=False).encode("utf-8"),
    file_name="route_roads_landmarks.csv",
    mime="text/csv",
)

st.markdown(
    f"**LocationIQ calls:** {ok_calls} OK · {throttled} throttled · {errs} errors · Total: {tot_calls}."
)
st.info("Tip: If throttled > 0, increase 'Delay between LocationIQ calls' and/or reduce 'Max sampled points per route'.")
