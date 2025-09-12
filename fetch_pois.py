# pages/fetch_pois.py
# Build: v2025.08.15-POI-1 (Standalone POI fetcher via LocationIQ; no Sheets writeback)

import io
import math
import re
import time
import urllib.parse
import pandas as pd
import requests
import streamlit as st

BUILD_ID = "v2025.08.15-POI-1"

st.set_page_config(page_title="Route POIs — LocationIQ", page_icon="📍", layout="wide")
st.title("📍 Route POIs — LocationIQ")
st.caption(f"Build: {BUILD_ID}")

# ----------------------------- Secrets --------------------------------------
LOCATIONIQ_API_KEY = None
try:
    LOCATIONIQ_API_KEY = st.secrets["locationiq"]["api_key"]
except Exception:
    pass

if not LOCATIONIQ_API_KEY:
    st.warning("LocationIQ API key not found in secrets. Add it under [locationiq] api_key = "..." to enable POIs.")

# ----------------------------- Utilities ------------------------------------
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

def load_from_excel_bytes(bts):
    xls = pd.ExcelFile(io.BytesIO(bts))
    dfs = {}
    if "Schedule" in xls.sheet_names:
        dfs["Schedule"] = pd.read_excel(xls, "Schedule", dtype=str)
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

def extract_route_id(u: str, source_id: str) -> str:
    rid = extract_route_id_from_url(u)
    if rid:
        return rid
    return source_id_to_digits(source_id)

# Strava API helpers (token should already be set from your OAuth page)
def strava_get_route(route_id: str, token: str):
    url = f"https://www.strava.com/api/v3/routes/{route_id}"
    r = requests.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=20)
    return r

# Polyline decoding (Google Encoded Polyline Algorithm)
def decode_polyline(polyline_str):
    if not polyline_str:
        return []
    index, lat, lng = 0, 0, 0
    coordinates = []
    length = len(polyline_str)

    while index < length:
        result, shift = 0, 0
        while True:
            if index >= length: break
            b = ord(polyline_str[index]) - 63
            index += 1
            result |= (b & 0x1f) << shift
            shift += 5
            if b < 0x20: break
        dlat = ~(result >> 1) if (result & 1) else (result >> 1)
        lat += dlat

        result, shift = 0, 0
        while True:
            if index >= length: break
            b = ord(polyline_str[index]) - 63
            index += 1
            result |= (b & 0x1f) << shift
            shift += 5
            if b < 0x20: break
        dlng = ~(result >> 1) if (result & 1) else (result >> 1)
        lng += dlng

        coordinates.append((lat / 1e5, lng / 1e5))
    return coordinates

# Haversine distance in meters
def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dl/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def sample_polyline(coords, interval_m=400, max_points=80):
    """Return a list of (lat, lon) sampled roughly every interval_m along the polyline."""
    if not coords:
        return []
    samples = [coords[0]]
    if len(coords) == 1:
        return samples
    acc = 0.0
    last = coords[0]
    for p in coords[1:]:
        d = haversine_m(last[0], last[1], p[0], p[1])
        if acc + d >= interval_m:
            samples.append(p)
            acc = 0.0
            last = p
        else:
            acc += d
            last = p
        if len(samples) >= max_points:
            break
    if samples[-1] != coords[-1] and len(samples) < max_points:
        samples.append(coords[-1])
    return samples

# LocationIQ reverse geocoding
def reverse_geocode_liq(lat, lon, api_key, timeout=15):
    url = "https://us1.locationiq.com/v1/reverse"
    params = {
        "key": api_key,
        "lat": f"{lat:.6f}",
        "lon": f"{lon:.6f}",
        "format": "json",
        "normalizecity": 1,
        "addressdetails": 1
    }
    r = requests.get(url, params=params, timeout=timeout)
    if r.status_code == 200:
        return r.json()
    return None

# in-session caches
if "poi_cache" not in st.session_state:
    st.session_state["poi_cache"] = {}
if "route_json_cache" not in st.session_state:
    st.session_state["route_json_cache"] = {}

# ---------------------------- Load Data UI ----------------------------------
mode = st.radio("Load data from:", ["Google Sheet (CSV)", "Upload Excel (.xlsx)"], horizontal=True)

dfs = None
if mode.startswith("Google"):
    url = st.text_input("Google Sheet URL", value="https://docs.google.com/spreadsheets/d/1ncT1NCbSnFsAokyFBkMWBVsk7yrJTiUfG0iBRxyUCTw/edit?usp=sharing", disabled=True)
    if url:
        try:
            dfs = load_from_google_csv(url)
        except Exception as e:
            st.error(f"Could not read Google Sheet: {e}")
else:
    up = st.file_uploader("Upload master Excel (.xlsx)", type=["xlsx"])
    if up:
        try:
            dfs = load_from_excel_bytes(up.read())
        except Exception as e:
            st.error(f"Could not read Excel: {e}")

if not dfs or "Schedule" not in dfs:
    st.error("Could not load a 'Schedule' tab.")
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

# Flatten rows
rows_long = []
synth_count = 0
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
        # synthesize URL from Source ID when URL is missing and Link Type mentions Strava
        if (not url) and ("strava" in lt.lower() or "strava" in nm.lower()):
            sid_digits = source_id_to_digits(sid_raw)
            if len(sid_digits) >= 6:
                url = f"https://www.strava.com/routes/{sid_digits}"
                synth_count += 1
        rid = extract_route_id(url, sid_raw) if ("strava" in lt.lower() or is_strava_route_url(url)) else ""
        rows_long.append({"Date": d, "Route Name": nm, "Side": side, "Link Type": lt, "URL": url, "Route ID": rid})

links_df = pd.DataFrame(rows_long)
st.write(f"Found {len(links_df)} route entries. (Synthesized {synth_count} Strava URLs from Source IDs)")

# Controls
st.subheader("POI Fetch Settings")
token = st.session_state.get("strava_token")
if not token:
    st.info("Connect your Strava account on the OAuth page first (needs read/read_all).")
start_at = st.number_input("Start at row", min_value=0, max_value=max(0, len(links_df)-1), value=0, step=1)
limit = st.slider("Rows in this pass", min_value=10, max_value=len(links_df), value=min(40, len(links_df)), step=10)
interval_m = st.slider("Sampling interval along route (meters)", min_value=100, max_value=1000, value=400, step=50)
max_points = st.slider("Max reverse-geocodes per route", min_value=10, max_value=120, value=60, step=5)
per_point_delay = st.slider("Delay between LocationIQ calls (seconds)", min_value=0.0, max_value=1.0, value=0.25, step=0.05)
per_route_delay = st.slider("Delay between routes (seconds)", min_value=0.0, max_value=2.0, value=0.2, step=0.1)

subset = links_df.iloc[start_at:start_at+limit].copy()

# Processing
out_rows = []
hit_limit = False
prog = st.progress(0, text="Fetching POIs...")

def cache_key(lat, lon):
    # round to 5 decimals (~1 meter) to improve cache hit-rate along nearby points
    return f"{lat:.5f},{lon:.5f}"

for i, (_, r) in enumerate(subset.iterrows(), start=1):
    url = r["URL"]; rid = r["Route ID"]; name = r["Route Name"]
    status = None
    poi_names = []
    categories = {}

    if not url or not is_strava_route_url(url):
        status = "Non-Strava or missing URL"
    elif not rid:
        status = "Missing route id"
    elif not token:
        status = "Needs Strava login"
    elif not LOCATIONIQ_API_KEY:
        status = "Missing LocationIQ key"
    else:
        # get strava route JSON (cached)
        rcache = st.session_state["route_json_cache"]
        data = rcache.get(rid)
        if not data:
            resp = strava_get_route(rid, token)
            if resp.status_code == 200:
                data = resp.json()
                rcache[rid] = data
            elif resp.status_code == 429:
                status = "Strava rate-limited (429)"; hit_limit = True
            elif resp.status_code == 404:
                status = "Route not found/accessible"
            elif resp.status_code == 401:
                status = "Unauthorized (Strava token/scope)"
            else:
                status = f"Strava API error {resp.status_code}"
        if data and not status:
            poly = None
            try:
                poly = data.get("map", {}).get("summary_polyline")
            except Exception:
                poly = None
            coords = decode_polyline(poly) if poly else []
            if not coords:
                status = "No polyline available"
            else:
                pts = sample_polyline(coords, interval_m=interval_m, max_points=max_points)
                # reverse for each sampled point with cache
                pcache = st.session_state["poi_cache"]
                for (lat, lon) in pts:
                    key = cache_key(lat, lon)
                    info = pcache.get(key)
                    if not info:
                        try:
                            info = reverse_geocode_liq(lat, lon, LOCATIONIQ_API_KEY)
                            if info:
                                pcache[key] = info
                        except Exception:
                            info = None
                        if per_point_delay > 0:
                            time.sleep(per_point_delay)
                    if info:
                        name_disp = info.get("display_name", "")
                        # try short name from address first
                        short = info.get("address", {}).get("neighbourhood") or info.get("address", {}).get("suburb") or info.get("address", {}).get("park") or ""
                        label = short or name_disp.split(",")[0]
                        label = label.strip()
                        if label and label not in poi_names:
                            poi_names.append(label)
                        cls = info.get("class") or info.get("category") or ""
                        typ = info.get("type") or ""
                        cat_key = f"{cls}:{typ}" if typ else cls
                        if cat_key:
                            categories[cat_key] = categories.get(cat_key, 0) + 1
                status = "OK" if poi_names else "No POIs found"

        if per_route_delay > 0 and status and status.startswith(("OK","No POIs")):
            time.sleep(per_route_delay)

    # Build summaries
    top_cats = ", ".join([f"{k}({v})" for k, v in sorted(categories.items(), key=lambda kv: kv[1], reverse=True)[:8]])
    summary = "; ".join(poi_names[:12])

    out_rows.append({
        "Date": r["Date"],
        "Route Name": name,
        "URL": url,
        "Route ID": rid,
        "Status": status or "Skipped",
        "POI Summary": summary,
        "Top Categories": top_cats,
        "Samples Used": len(poi_names)
    })

    prog.progress(min(i/len(subset), 1.0))

    if hit_limit:
        st.warning("Hit Strava 429 limit — pause and resume later with a smaller batch or higher delay.")
        break

prog.empty()

poi_df = pd.DataFrame(out_rows)
st.dataframe(poi_df, use_container_width=True, hide_index=True)

st.download_button(
    "⬇️ Download POIs (CSV)",
    data=poi_df.to_csv(index=False).encode("utf-8"),
    file_name="route_pois.csv",
    mime="text/csv",
)

st.info("Tip: Increase 'Sampling interval' or lower 'Max reverse-geocodes per route' if you hit LocationIQ throttling. Add your LocationIQ key in secrets to enable results.")
