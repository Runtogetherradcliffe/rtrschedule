
# pages/social_posts.py
# Build: v2025.08.16-SOCIAL-1c (ASCII-clean, quoting fixed)

import io
import re
import time
import urllib.parse
import pandas as pd
import requests
import streamlit as st
from math import radians, sin, cos, asin, sqrt
from datetime import datetime

st.set_page_config(page_title="Weekly Social Post Composer", page_icon=":mega:", layout="wide")
st.title("Weekly Social Post Composer")
st.caption("Build: v2025.08.16-SOCIAL-1c - 2 routes per week from your Google Sheet, with Strava links + distance/elevation + POIs.")

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

def load_schedule_gsheet(url):
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
    dlon = radians(lat2-lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return R * c

def sample_polyline(coords, step_m=700, max_points=24):
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
        "zoom": 18,
    }
    return requests.get(u, params=params, timeout=25)

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

def minimal_poi_from_polyline(polyline_str, liq_key, liq_delay=0.7, step_m=700, max_pts=24):
    if not polyline_str:
        return [], []
    coords = decode_polyline(polyline_str)
    pts = sample_polyline(coords, step_m=step_m, max_points=max_pts)
    roads = set()
    marks = set()
    for (lat, lon) in pts:
        try:
            rr = locationiq_reverse(lat, lon, liq_key)
            if rr.status_code == 200:
                p = rr.json()
                for rn in extract_road_names(p):
                    roads.add(rn)
                if is_allowed_landmark(p):
                    nm = p.get("name") or p.get("display_name")
                    if nm:
                        marks.add(nm)
            time.sleep(liq_delay)
        except Exception:
            pass
    return sorted(roads), sorted(marks)

def try_float(s):
    try:
        return float(s)
    except Exception:
        return None

# --------------------------- Settings (sheet URL memory) -----------------
st.sidebar.header("Settings")
gs_default = st.sidebar.text_input("Default Google Sheet URL", value=st.session_state.get("GS_URL_DEFAULT", ""))
if st.sidebar.button("Save as session default"):
    st.session_state["GS_URL_DEFAULT"] = gs_default
    st.sidebar.success("Saved for this session.")

# ---------------------------- Input controls -----------------------------
st.subheader("Pick a week")
gs_url = st.text_input("Google Sheet URL", value=st.session_state.get("GS_URL_DEFAULT", ""))
date_hint = st.text_input("Week date (YYYY-MM-DD) - any date in the week (or leave blank for next upcoming in sheet)", value="")

if not gs_url:
    st.stop()

dfs = load_schedule_gsheet(gs_url)
if "Schedule" not in dfs:
    st.error("Could not load Schedule tab from the Google Sheet.")
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
r_terrain = [find_col(["route1terrain","route1terraintype","route1terrainroadtrailmixed"]), find_col(["route2terrain","route2terraintype","route2terrainroadtrailmixed"])]
r_area = [find_col(["route1area"]), find_col(["route2area"])]
# Optional metadata columns already in your sheet (if present)
r_dist = [find_col(["route1distance","route1distancekm","route1distkm"]), find_col(["route2distance","route2distancekm","route2distkm"])]
r_elev = [find_col(["route1elevation","route1elevationgain","route1elevationgainm"]), find_col(["route2elevation","route2elevationgain","route2elevationgainm"])]
r_pois = [find_col(["route1pois","route1poissummary","roads(on-route)"]), find_col(["route2pois","route2poissummary","roads(on-route)"])]

if not date_col or not all(r_names) or not all(r_urls):
    st.error("Missing required columns in Schedule (Date, Route Names, Route URLs).")
    st.stop()

# Choose the row for the requested week
target_idx = None
if date_hint:
    try:
        dt = pd.to_datetime(date_hint).date()
        sched["_dateparsed"] = pd.to_datetime(sched[date_col], errors="coerce")
        matches = sched[sched["_dateparsed"].dt.date == dt]
        if not matches.empty:
            target_idx = matches.index[0]
        else:
            later = sched[sched["_dateparsed"].dt.date >= dt]
            if not later.empty:
                target_idx = later.index[0]
    except Exception:
        target_idx = None

if target_idx is None:
    try:
        sched["_dateparsed"] = pd.to_datetime(sched[date_col], errors="coerce")
        today = pd.Timestamp.utcnow().normalize()
        later = sched[sched["_dateparsed"] >= today]
        target_idx = later.index[0] if not later.empty else sched.index[0]
    except Exception:
        target_idx = sched.index[0]

row = sched.loc[target_idx]

def build_route_dict(side_idx: int):
    nm = clean(row.get(r_names[side_idx], ""))
    lt = clean(row.get(r_types[side_idx], "")) if r_types[side_idx] else ""
    url_raw = clean(row.get(r_urls[side_idx], ""))
    sid_raw = clean(row.get(r_srcid[side_idx], "")) if r_srcid[side_idx] else ""
    terr = clean(row.get(r_terrain[side_idx], "")) if r_terrain[side_idx] else ""
    area = clean(row.get(r_area[side_idx], "")) if r_area[side_idx] else ""
    url = make_https(url_raw)
    rid = extract_route_id(url, sid_raw) if ("strava" in lt.lower() or is_strava_route_url(url)) else ""
    dist = try_float(row.get(r_dist[side_idx], "")) if r_dist[side_idx] else None
    elev = try_float(row.get(r_elev[side_idx], "")) if r_elev[side_idx] else None
    pois = clean(row.get(r_pois[side_idx], "")) if r_pois[side_idx] else ""
    return {"name": nm, "link_type": lt, "url": url, "rid": rid, "terrain": terr, "area": area,
            "dist": dist, "elev": elev, "pois": pois}

routes = [build_route_dict(0), build_route_dict(1)]

st.subheader("Selected Week")
st.write(f"Date: {clean(row.get(date_col))}")
colA, colB = st.columns(2)
with colA: st.write(f"Route 1: {routes[0]['name']}")
with colB: st.write(f"Route 2: {routes[1]['name']}")

# ----------------------- Fetch missing bits (distance/elev/POIs) ---------
access_token = st.session_state.get("strava_token")
liq_key = None
try:
    liq_key = st.secrets["locationiq"]["api_key"]
except Exception:
    liq_key = None

fetch_missing = st.checkbox("Fetch missing distance/elevation/POIs if not present", value=True)
liq_delay = st.slider("LocationIQ delay (s)", 0.2, 1.0, 0.7, 0.05, help="Only used when fetching missing POIs.")
if fetch_missing:
    for r in routes:
        # Strava distance/elevation if missing
        if (r["dist"] is None or r["elev"] is None) and access_token and r["rid"]:
            resp = strava_get_route_json(r["rid"], access_token)
            if resp.status_code == 200:
                data = resp.json()
                dist_m = data.get("distance")
                elev_m = data.get("elevation_gain") or data.get("elevation_gain_total") or data.get("elevation")
                try:
                    if dist_m is not None and r["dist"] is None:
                        r["dist"] = round(float(dist_m)/1000.0, 2)
                except Exception:
                    pass
                try:
                    if elev_m is not None and r["elev"] is None:
                        r["elev"] = round(float(elev_m), 1)
                except Exception:
                    pass
                # Light POI pass if needed
                if (not r["pois"]) and liq_key:
                    poly = (data.get("map") or {}).get("polyline") or (data.get("map") or {}).get("summary_polyline")
                    if poly:
                        coords = decode_polyline(poly)
                        pts = sample_polyline(coords, step_m=900, max_points=16)
                        roads = set(); marks = set()
                        for (lat, lon) in pts:
                            try:
                                rr = locationiq_reverse(lat, lon, liq_key)
                                if rr.status_code == 200:
                                    p = rr.json()
                                    for rn in extract_road_names(p): roads.add(rn)
                                    if is_allowed_landmark(p):
                                        nm = p.get("name") or p.get("display_name")
                                        if nm: marks.add(nm)
                                time.sleep(liq_delay)
                            except Exception:
                                pass
                        parts = []
                        if roads: parts.append("; ".join(sorted(list(roads))[:6]))
                        if marks: parts.append("; ".join(sorted(list(marks))[:4]))
                        r["pois"] = " | ".join(parts)

# ----------------------------- Compose posts ------------------------------
st.subheader("Compose")
meet_loc_col = None
for c in sched.columns:
    if "meet" in c.lower() and "loc" in c.lower():
        meet_loc_col = c
        break
meet_loc = clean(row.get(meet_loc_col, "")) if meet_loc_col else ""

date_str = clean(row.get(date_col))
title_line = f"Thursday Run - {date_str}"
r1 = routes[0]
r2 = routes[1]

def fmt_route(r):
    bits = []
    if r["name"]: bits.append(r["name"])
    if r["dist"] is not None: bits.append(f"{r['dist']} km")
    if r["elev"] is not None: bits.append(f"+{r['elev']} m")
    if r["terrain"]: bits.append(r["terrain"])
    if r["area"]: bits.append(r["area"])
    head = " - ".join(bits)
    tail = []
    if r["pois"]: tail.append(r["pois"])
    if r["url"]: tail.append(r["url"])
    return head + ("\n" + "\n".join(tail) if tail else "")

long_post = '''Thursday Run - {date_str}

Route A: {r1}

Route B: {r2}

Meet: {meet} at 7pm
All paces welcome. See you there!'''.format(
    date_str=date_str, r1=fmt_route(r1), r2=fmt_route(r2), meet=(meet_loc or "Radcliffe Market")
)

short_post = '{date}: {n1} ({d1}km) & {n2} ({d2}km). Meet {meet} 7pm. {u1} {u2}'.format(
    date=date_str,
    n1=r1['name'], d1=(r1['dist'] or ''), n2=r2['name'], d2=(r2['dist'] or ''),
    meet=(meet_loc or 'Radcliffe Market'),
    u1=(r1['url'] or ''), u2=(r2['url'] or '')
)

st.text_area("Facebook/Instagram caption", long_post, height=260)
st.text_area("X / Twitter", short_post, height=120)

# Preview card
st.subheader("Preview")
st.markdown(
    "**{ttl}**  \n- **Route 1:** {n1} - {d1} km, +{e1} m ({t1})  \n- **Route 2:** {n2} - {d2} km, +{e2} m ({t2})  \n**Meet:** {meet} at 7pm  ".format(
        ttl=title_line,
        n1=r1['name'], d1=(r1['dist'] or '?'), e1=(r1['elev'] or '?'), t1=(r1['terrain'] or 'Terrain ?'),
        n2=r2['name'], d2=(r2['dist'] or '?'), e2=(r2['elev'] or '?'), t2=(r2['terrain'] or 'Terrain ?'),
        meet=(meet_loc or 'Radcliffe Market')
    )
)
