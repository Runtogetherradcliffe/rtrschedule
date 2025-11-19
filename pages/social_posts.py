
# pages/social_posts.py
# Build: v2025.09.01-SOCIAL-24 (rebuild: robust date-only + Strava/LocationIQ enrichment)

import re
import random
import urllib.parse
from datetime import datetime

SAFETY_NOTE = "As we are now running after dark, please remember lights and hi-viz, be safe, be seen!"
import pandas as pd
import requests
import os
import hashlib
import streamlit as st
import streamlit.components.v1 as components
from radar import radar_match_steps
from roads_integration import must_include_from_sheet

# Debug container to avoid NameError even if later code fails to set it
poi_debug: list = []
from typing import List, Dict

# ----------------------------
# Config / helpers
# ----------------------------
def get_cfg(key, default=None):
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default

def clean(s):
    if s is None:
        return ""
    s = str(s).strip()
    # Treat common pandas null markers as empty
    if s.lower() in ("nan", "nat"):
        return ""
    return s

def try_float(x, default=None):
    """Safely convert a value to float, returning default if conversion fails."""
    try:
        if x is None or x == "":
            return default
        return float(x)
    except (TypeError, ValueError):
        return default


def make_https(url):
    """Ensure any non-empty URL uses https:// scheme."""
    if not url:
        return ""
    u = str(url).strip()
    if u.startswith("http://"):
        u = "https://" + u[len("http://"):]
    return u


def ordinal(n:int)->str:
    n=int(n)
    return f"{n}{'th' if 11<=n%100<=13 else {1:'st',2:'nd',3:'rd'}.get(n%10,'th')}"

def format_day_month_uk(dts):
    ts = pd.to_datetime(dts)
    return f"{ordinal(ts.day)} {ts.strftime('%B')}"

def hilliness_blurb(dist_km, elev_m):
    phrases = {
        "flat": ["flat and friendly 🏁","fast & flat 🏁","pan-flat cruise 💨"],
        "rolling": ["gently rolling 🌱","undulating and friendly 🌿","rolling countryside vibes 🌳"],
        "hilly": ["a hilly tester! ⛰️","spicy climbs ahead 🌶️","some punchy hills 🚵"],
    }
    if not dist_km or not elev_m:
        return random.choice(["a great midweek spin","perfect for all paces","midweek miles made easy"])
    try:
        m_per_km = float(elev_m)/max(float(dist_km),0.1)
    except Exception:
        return random.choice(["a great midweek spin","perfect for all paces","midweek miles made easy"])
    key = "flat" if m_per_km < 10 else ("rolling" if m_per_km < 20 else "hilly")
    return random.choice(phrases[key])

# ----------------------------
# OAuth / APIs
# ----------------------------
def _get_secret(name, default=None):
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default

def get_strava_token():
    for k in ("strava_access_token","strava_token","access_token"):
        tok = st.session_state.get(k)
        if tok:
            return tok
    cid  = _get_secret("STRAVA_CLIENT_ID")
    csec = _get_secret("STRAVA_CLIENT_SECRET")
    rtok = st.session_state.get("strava_refresh_token") or _get_secret("STRAVA_REFRESH_TOKEN")
    if cid and csec and rtok:
        try:
            r = requests.post("https://www.strava.com/oauth/token", data={
                "client_id":cid, "client_secret":csec, "grant_type":"refresh_token", "refresh_token":rtok
            }, timeout=15)
            if r.ok:
                j = r.json()
                st.session_state["strava_access_token"] = j.get("access_token")
                st.session_state["strava_refresh_token"] = j.get("refresh_token", rtok)
                return j.get("access_token")
        except Exception:
            pass
    return None

def _extract_route_id(url_or_id):
    if not url_or_id: return None
    s = str(url_or_id)
    m = re.search(r"/routes/(\d+)", s)
    if m: return m.group(1)
    m = re.search(r"(\d{6,})", s)
    return m.group(1) if m else None

def fetch_strava_route_metrics(url_or_id):
    rid = _extract_route_id(url_or_id)
    tok = get_strava_token()
    if not rid or not tok:
        return None
    try:
        r = requests.get(f"https://www.strava.com/api/v3/routes/{rid}",
                         headers={"Authorization": f"Bearer {tok}"}, timeout=15)
        if not r.ok:
            return None
        j = r.json()
        return {
            "dist_km": (float(j["distance"])/1000.0) if j.get("distance") is not None else None,
            "elev_m": float(j["elevation_gain"]) if j.get("elevation_gain") is not None else None,
            "polyline": (j.get("map", {}) or {}).get("polyline") or (j.get("map", {}) or {}).get("summary_polyline")
        }
    except Exception:
        return None

# --- GPX fallback for distance/elevation and POIs if no polyline ---
import math
import xml.etree.ElementTree as ET

def _haversine(lat1, lon1, lat2, lon2):
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl   = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dl/2)**2
    return 2*R*math.asin(math.sqrt(a))

def _parse_gpx(gpx_bytes):
    try:
        root = ET.fromstring(gpx_bytes)
    except Exception:
        return []
    ns = {"gpx":"http://www.topografix.com/GPX/1/1"}
    pts = []
    for trkpt in root.findall(".//gpx:trkpt", ns):
        lat = trkpt.get("lat"); lon = trkpt.get("lon")
        if lat is None or lon is None: continue
        ele_el = trkpt.find("gpx:ele", ns)
        ele = float(ele_el.text) if ele_el is not None else None
        pts.append((float(lat), float(lon), ele))
    return pts

def _metrics_from_points(pts):
    if len(pts) < 2: return None, None
    dist_m, gain = 0.0, 0.0
    prev = pts[0]
    for cur in pts[1:]:
        dist_m += _haversine(prev[0], prev[1], cur[0], cur[1])
        if prev[2] is not None and cur[2] is not None:
            d = cur[2] - prev[2]
            if d > 0: gain += d
        prev = cur
    return dist_m/1000.0, gain

def fetch_strava_route_gpx_points(route_id, token):
    try:
        r = requests.get(f"https://www.strava.com/api/v3/routes/{route_id}/export_gpx",
                         headers={"Authorization": f"Bearer {token}"}, timeout=20)
        if not r.ok: return []
        return _parse_gpx(r.content)
    except Exception:
        return []

def _decode_polyline(polyline_str):
    if not polyline_str: return []
    index, lat, lng = 0, 0, 0
    coords = []
    while index < len(polyline_str):
        result, shift = 0, 0
        while True:
            b = ord(polyline_str[index]) - 63
            index += 1
            result |= (b & 0x1f) << shift
            shift += 5
            if b < 0x20: break
        dlat = ~(result >> 1) if (result & 1) else (result >> 1)
        lat += dlat
        result, shift = 0, 0
        while True:
            b = ord(polyline_str[index]) - 63
            index += 1
            result |= (b & 0x1f) << shift
            shift += 5
            if b < 0x20: break
        dlng = ~(result >> 1) if (result & 1) else (result >> 1)
        lng += dlng
        coords.append((lat/1e5, lng/1e5))
    return coords

def _sample_points(polyline, max_pts=10):
    pts = _decode_polyline(polyline)
    if not pts: return []
    if len(pts) <= max_pts: return pts
    step = max(1, len(pts)//max_pts)
    return pts[::step]






def get_locationiq_key(*, debug: bool = False):
    """
    Try a bunch of places/aliases for the LocationIQ API key so we match other pages:
    - st.secrets top-level: LOCATIONIQ_API_KEY / LOCATIONIQ_TOKEN / LOCATIONIQ_KEY / locationiq_api_key
    - st.secrets nested: [locationiq].api_key / token / key (no type assumption; treat like mapping)
    - st.session_state: same aliases
    - os.environ: LOCATIONIQ_API_KEY / LOCATIONIQ_TOKEN
    Returns key (and optionally a debug dict showing where it was found).
    """
    candidates = []
    # secrets (top-level)
    try:
        s = st.secrets
        for k in ["LOCATIONIQ_API_KEY","LOCATIONIQ_TOKEN","LOCATIONIQ_KEY","locationiq_api_key","locationiq_token","locationiq_key"]:
            try:
                v = s.get(k, None)
            except Exception:
                try:
                    v = s[k]
                except Exception:
                    v = None
            candidates.append(("secrets", k, v))
        # secrets (nested sections) without assuming dict type
        for sect in ["locationiq","LOCATIONIQ"]:
            try:
                sec = s[sect]
            except Exception:
                sec = None
            if sec is not None:
                for k in ["api_key","token","key"]:
                    try:
                        v = sec.get(k, None) if hasattr(sec, "get") else sec[k]
                    except Exception:
                        v = None
                    candidates.append((f"secrets[{sect}]", k, v))
    except Exception:
        pass
    # session_state
    try:
        for k in ["LOCATIONIQ_API_KEY","LOCATIONIQ_TOKEN","LOCATIONIQ_KEY","locationiq_api_key","locationiq_token","locationiq_key"]:
            v = st.session_state.get(k, None)
            candidates.append(("session_state", k, v))
        if "locationiq" in st.session_state and isinstance(st.session_state["locationiq"], dict):
            for k in ["api_key","token","key"]:
                v = st.session_state["locationiq"].get(k, None)
                candidates.append(("session_state[locationiq]", k, v))
    except Exception:
        pass
    # environment
    try:
        import os as _os
        for k in ["LOCATIONIQ_API_KEY","LOCATIONIQ_TOKEN","LOCATIONIQ_KEY"]:
            v = _os.environ.get(k, None)
            candidates.append(("env", k, v))
    except Exception:
        pass

    # choose first truthy
    for origin, name, val in candidates:
        if val:
            return (str(val), {"source": origin, "name": name}) if debug else str(val)
    return (None, {"source": None, "name": None}) if debug else None







def locationiq_pois(polyline=None, sample_points=None, *, rid: str | None = None, debug: bool = False):
    """
    POI highlighting disabled for social posts – we no longer look up on-route features.
    Kept as a lightweight stub for compatibility.
    """
    if debug:
        return [], {"note": "poi_lookup_disabled"}
    return []

def enrich_route_dict(r: dict) -> dict:
    """Return a new dict with dist/elev/pois populated from Strava + LocationIQ.
       Always prefer Strava distance over sheet placeholders."""
    rid = _extract_route_id(r.get("url") or r.get("rid"))
    tok = get_strava_token()
    dist_km = r.get("dist")
    elev_m  = r.get("elev")
    polyline = None
    sample = []

    if rid and tok:
        meta = fetch_strava_route_metrics(r.get("url") or r.get("rid"))
        if meta:
            if meta.get("dist_km") is not None:
                dist_km = meta["dist_km"]
            if elev_m is None and meta.get("elev_m") is not None:
                elev_m = meta["elev_m"]
            polyline = meta.get("polyline")

        if dist_km is None or elev_m is None or not polyline:
            pts = fetch_strava_route_gpx_points(rid, tok)
            if pts:
                d_km, g_m = _metrics_from_points(pts)
                if dist_km is None and d_km is not None:
                    dist_km = d_km
                if elev_m is None and g_m is not None:
                    elev_m = g_m
                if not polyline:
                    step = max(1, len(pts)//40)
                    sample = [(lat, lon) for (lat,lon,ele) in pts[::step]]

    pois = r.get("pois")
    if not pois:
        pois_list, _poi_rep = locationiq_pois(polyline, sample_points=sample, rid=rid, debug=True)
        if pois_list:
            pois = ", ".join(pois_list)

    out = dict(r)
    out["dist"] = dist_km
    out["polyline"] = polyline
    if elev_m is not None: out["elev"] = elev_m
    if pois: out["pois"] = pois
    return out

# ----------------------------
# UI: Sheet URL + load
# ----------------------------
st.title("Weekly Social Post Composer")
st.caption("Build: v2025.09.01-SOCIAL-24 — date-only parsing, Strava/LocationIQ enrichment")

sheet_url = st.text_input("Google Sheet URL", value="https://docs.google.com/spreadsheets/d/1ncT1NCbSnFsAokyFBkMWBVsk7yrJTiUfG0iBRxyUCTw/edit?usp=sharing", disabled=True)
if not sheet_url:
    st.info("Paste your Google Sheet (the master schedule).")
    st.stop()


import urllib.parse

def csv_url_candidates(url: str, sheet_name: str = "Schedule"):
    """Return a list of candidate CSV endpoints for a Google Sheet link."""
    cands = []
    if not url:
        return cands

    # Keep original first if it's already csv/gviz
    if "format=csv" in url or "/gviz/tq" in url:
        cands.append(url)

    # Parse parts
    u = urllib.parse.urlparse(url)
    # spreadsheet id
    sid = None
    m = re.search(r"/spreadsheets/d/([^/]+)", u.path)
    if m:
        sid = m.group(1)

    # Try to find gid in query or fragment
    q = urllib.parse.parse_qs(u.query)
    gid = (q.get("gid", [None])[0]) or (u.fragment.split("gid=")[-1] if "gid=" in u.fragment else None)

    if sid and gid:
        cands.append(f"https://docs.google.com/spreadsheets/d/{sid}/export?format=csv&gid={gid}")

    if sid:
        # gviz by sheet name
        cands.append(
            f"https://docs.google.com/spreadsheets/d/{sid}/gviz/tq"
            f"?tqx=out:csv&sheet={urllib.parse.quote(sheet_name)}"
        )

    # Deduplicate while preserving order
    seen = set(); uniq = []
    for cu in cands:
        if cu not in seen:
            seen.add(cu); uniq.append(cu)
    return uniq

# Build candidates and try each
csv_candidates = csv_url_candidates(sheet_url, sheet_name=get_cfg("SHEET_NAME","Schedule"))
sched = None
err = None
for cu in csv_candidates:
    try:
        sched = pd.read_csv(cu)
        break
    except Exception as e:
        err = e
if sched is None:
    st.error("Couldn't read the Google Sheet as CSV. Is it shared correctly?")
    st.stop()


# Column mapping (as per your master sheet)
date_col = "Date (Thu)"
r_names = ["Route 1 - Name", "Route 2 - Name"]
r_urls  = ["Route 1 - Route Link (Source URL)", "Route 2 - Route Link (Source URL)"]
r_srcid = ["Route 1 - Source ID", "Route 2 - Source ID"]
r_terrain = ["Route 1 - Terrain (Road/Trail/Mixed)", "Route 2 - Terrain (Road/Trail/Mixed)"]
r_area    = ["Route 1 - Area", "Route 2 - Area"]
r_dist    = ["Route 1 - Distance (km)", "Route 2 - Distance (km)"]
# Elevation/POI columns not present in your sheet
r_elev = [None, None]
r_pois = [None, None]
notes_col = "Notes"
meet_loc_col = None  # not in sheet; parsed from Notes

# Route 3 (e.g. social walk / C25K) and optional On Tour map link
route3_desc_col = "Route 3 description"
route3_name_col = "Route 3 name"
route3_dist_col = "Route 3 distance"
route3_url_col  = "Route 3 URL"
on_tour_map_col = "On tour meeting location map"

# --- Parse dates as pure date ---
d = pd.to_datetime(sched[date_col], errors="coerce", format="%Y-%m-%d %H:%M:%S")
sched["_dateonly"] = d.dt.date
today = pd.Timestamp.today().date()
future_rows = sched[sched["_dateonly"] >= today]
future_rows = future_rows[pd.to_datetime(future_rows["_dateonly"]).dt.weekday == 3]  # Thu
if future_rows.empty:
    st.warning("No upcoming Thursday entries found.")
    st.stop()
future_rows = future_rows.sort_values("_dateonly")

opt_idx = future_rows.index.tolist()
def _fmt(idx):
    return format_day_month_uk(pd.to_datetime(future_rows.loc[idx, "_dateonly"]))
idx_choice = st.selectbox("Date", options=opt_idx, format_func=_fmt, index=0 if opt_idx else None)
if idx_choice is None:
    st.stop()

row = future_rows.loc[idx_choice]

# Toggle Jeffing option (some weeks there may be no Jeffing group, e.g. during C25K)
show_jeffing = st.checkbox(
    "Include Jeffing option in messages",
    value=True,
    help="Untick this if there will be no Jeffing group this week.",
)

# ----------------------------
# Build route dicts (sheet)
# ----------------------------
def route_id(source_id_raw, url):
    rid = _extract_route_id(url)
    if rid: return rid
    return str(source_id_raw).strip() if source_id_raw else ""

def build_route_dict(side_idx: int) -> dict:
    nm = clean(row.get(r_names[side_idx], ""))
    url_raw = clean(row.get(r_urls[side_idx], ""))
    url = make_https(url_raw)
    rid = route_id(row.get(r_srcid[side_idx], ""), url)
    terr = clean(row.get(r_terrain[side_idx], ""))
    area = clean(row.get(r_area[side_idx], ""))
    dist = try_float(row.get(r_dist[side_idx], ""))
    elev = None
    pois = ""
    return {"name": nm, "url": url, "rid": rid, "terrain": terr, "area": area,
            "dist": dist, "elev": elev, "pois": pois}

routes = [build_route_dict(0), build_route_dict(1)]

# Enrich from Strava + LocationIQ (prefer Strava distance)

routes = [enrich_route_dict(r) for r in routes]

# Optional Route 3 (e.g. social walk / C25K)
route3 = None
route3_desc = clean(row.get(route3_desc_col, ""))
try:
    route3_name = clean(row.get(route3_name_col, ""))
    route3_url_raw = clean(row.get(route3_url_col, ""))
    if route3_name or route3_url_raw or route3_desc:
        route3_url = make_https(route3_url_raw)
        route3_rid = route_id("", route3_url) if route3_url else ""
        route3_dist_val = try_float(row.get(route3_dist_col, ""))
        route3 = {
            "name": route3_name or (route3_desc or "Route 3"),
            "url": route3_url,
            "rid": route3_rid,
            "terrain": "",
            "area": "",
            "dist": route3_dist_val,
            "elev": None,
            "pois": "",
        }
        route3 = enrich_route_dict(route3)
except Exception:
    route3 = None

# Determine if tonight is a Road run (from row/routing terrain)
try:
    is_road = any(((r.get("terrain") or "").lower().startswith("road") or "road" in (r.get("terrain") or "").lower()) for r in routes)
    if not is_road:
        # Fallback: inspect the sheet row for any 'terrain'/'type' columns
        for k in row.index:
            v = str(row.get(k, "")).lower()
            if "road" in v and ("terrain" in k.lower() or "type" in k.lower() or "season" in k.lower()):
                is_road = True
                break
except Exception:
    is_road = False


poi_debug = []
try:
    m = st.session_state.get("poi_debug_map", {})
    for r in routes:
        rid = r.get("rid")
        if rid and rid in m:
            rep = dict(m[rid])
            rep["rid"] = rid
            poi_debug.append(rep)
except Exception:
    poi_debug = poi_debug  # no-op




# --- Ensure routes carry POIs into the message even if enrichment skipped setting them ---
try:
    if poi_debug:
        rid2pois = {}
        for rep in poi_debug:
            if isinstance(rep, dict) and rep.get("rid") and rep.get("final_pois"):
                rid2pois[rep["rid"]] = rep["final_pois"]
        for r in routes:
            if r.get("rid") and (not r.get("pois")) and r.get("rid") in rid2pois:
                r["pois"] = ", ".join(rid2pois[r["rid"]])
except Exception:
    pass

with st.expander("Debug: POIs (per-route)", expanded=False):
    dbg = globals().get("poi_debug", [])
    try:
        st.json(dbg if dbg else [{"note":"no debug collected"}])
    except Exception:
        st.write(dbg if dbg else [{"note":"no debug collected"}])

poi_debug = []
try:
    for r in routes:
        if r.get('rid'):
            _, rep = locationiq_pois(None, sample_points=_sample_points(r.get('polyline')) if r.get('polyline') else None, rid=r.get('rid'), debug=True)
            poi_debug.append(rep if isinstance(rep, dict) else {'rid': r.get('rid'), 'note':'no_rep'})
except Exception:
    pass



# ----------------------------
# Weather helper (Open-Meteo)
# ----------------------------
def get_weather_blurb_for_date(run_date):
    """Return a short clothing suggestion based on forecast temp around 7pm.
    Uses Open-Meteo with a fixed Radcliffe lat/lon; fails gracefully if anything goes wrong.
    """
    try:
        from datetime import datetime as _dt
        # Normalise run_date to a date object
        try:
            import pandas as _pd  # type: ignore
        except Exception:
            _pd = None
        d = None
        if _pd is not None and isinstance(run_date, (_pd.Timestamp,)):
            d = run_date.date()
        elif isinstance(run_date, (_dt,)):
            d = run_date.date()
        else:
            s = str(run_date)[:10]
            try:
                d = _dt.strptime(s, "%Y-%m-%d").date()
            except Exception:
                return None
        # Radcliffe approx coordinates
        lat, lon = 53.561, -2.329
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": "temperature_2m",
            "timezone": "Europe/London",
            "start_date": d.isoformat(),
            "end_date": d.isoformat(),
        }
        resp = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=8)
        if not resp.ok:
            return None
        data = resp.json()
        hourly = data.get("hourly", {})
        times = hourly.get("time") or []
        temps = hourly.get("temperature_2m") or []
        if not times or not temps:
            return None
        target = _dt.combine(d, _dt.min.time()).replace(hour=19)
        best_temp = None
        best_diff = None
        for t_str, temp in zip(times, temps):
            try:
                t_dt = _dt.fromisoformat(t_str)
            except Exception:
                continue
            diff = abs((t_dt - target).total_seconds())
            if best_diff is None or diff < best_diff:
                best_diff = diff
                best_temp = temp
        if best_temp is None:
            return None
        try:
            t = float(best_temp)
        except Exception:
            return None
        if t <= 5:
            return "It’s looking cold around session time – layer up, hats and gloves recommended ❄️"
        if t >= 22:
            return "It’s going to be a warm one – bring water and dress for the heat ☀️"
        return None
    except Exception:
        return None

# ----------------------------
# Compose message
# ----------------------------
# Meeting location: parse from Notes (pattern "Meeting: XYZ")
meet_loc = ""
if meet_loc_col:
    meet_loc = clean(row.get(meet_loc_col, ""))
if not meet_loc:
    notes = str(row.get(notes_col, ""))
    m = re.search(r"Meeting:\s*([^|\n]+)", notes, re.IGNORECASE)
    if m: meet_loc = m.group(1).strip()
if not meet_loc:
    meet_loc = get_cfg("MEET_LOC_DEFAULT", "Radcliffe Market")


meet_loc_stripped = meet_loc.strip()
is_on_tour_meeting = meet_loc_stripped.lower() not in ("", "radcliffe market")
try:
    meet_map_url = clean(row.get(on_tour_map_col, ""))
except Exception:
    meet_map_url = ""
nice_meet_loc = meet_loc_stripped or get_cfg("MEET_LOC_DEFAULT", "Radcliffe Market")

date_str = format_day_month_uk(pd.to_datetime(row["_dateonly"]))
time_line = "🕖 We set off at 7:00pm"
meeting_line = f"📍 Meeting at: {meet_loc.title()}"

def locationiq_match_steps(polyline: str | None, *, max_pts: int = 120):
    """Map-matching steps (maneuvers) via LocationIQ; tries foot→cycling→driving."""
    if not polyline:
        return []
    key = get_locationiq_key()
    if not key:
        return []
    base = _get_secret("LOCATIONIQ_BASE", "us1") or "us1"
    pts = _sample_points(polyline, max_pts=max_pts)
    if not pts:
        return []
    coords = ";".join([f"{lon:.6f},{lat:.6f}" for (lat, lon) in pts])
    profiles = ["foot", "cycling", "driving"]
    for prof in profiles:
        url = f"https://{base}.locationiq.com/v1/matching/{prof}/{coords}"
        try:
            r = requests.get(url, params={
                "key": key,
                "steps": "true",
                "geometries": "geojson",
                "overview": "false",
                "annotations": "false",
                "alternatives": "false",
            }, timeout=8)
            if not r.ok:
                continue
            j = r.json() or {}
            matchings = j.get("matchings") or []
            if not matchings:
                continue
            steps_out = []
            prev_name = None
            for leg in (matchings[0].get("legs") or []):
                for step in (leg.get("steps") or []):
                    nm = (step.get("name") or "").strip()
                    if not nm or nm.lower() == "unnamed road":
                        continue
                    coords = []
                    try:
                        for lon, lat in (step.get("geometry", {}).get("coordinates") or []):
                            coords.append((lat, lon))
                    except Exception:
                        pass
                    man = step.get("maneuver") or {}
                    if steps_out and prev_name and nm.lower() == prev_name.lower():
                        steps_out[-1]["coords"].extend(coords)
                    else:
                        steps_out.append({"name": nm, "coords": coords, "maneuver": man})
                    prev_name = nm
            if steps_out:
                return steps_out
        except Exception:
            continue
    return []

_REVERSE_CACHE = {}
def reverse_cache_lookup(lat: float, lon: float, *, zooms=(18,17,16)):
    k = (round(lat, 5), round(lon, 5))
    if k in _REVERSE_CACHE:
        return _REVERSE_CACHE[k]
    key = get_locationiq_key()
    if not key:
        return None
    base = _get_secret("LOCATIONIQ_BASE", "us1") or "us1"
    for z in zooms:
        try:
            r = requests.get(
                f"https://{base}.locationiq.com/v1/reverse",
                params={
                    "key": key, "lat": f"{lat:.6f}", "lon": f"{lon:.6f}",
                    "format": "json", "normalizeaddress": 1, "addressdetails": 1, "zoom": z,
                },
                timeout=2.5,
            )
            if not r.ok:
                continue
            js = r.json() or {}
            a = js.get("address") or {}
            for kname in ("road","pedestrian","footway","path","cycleway","residential"):
                v = a.get(kname)
                if v and str(v).strip().lower() != "unnamed road":
                    _REVERSE_CACHE[k] = str(v).strip()
                    return _REVERSE_CACHE[k]
        except Exception:
            continue
    _REVERSE_CACHE[k] = None
    return None

def onroute_named_segments(polyline: str, *, max_pts: int = 72):
    """On-route segments using cached reverse geocoding; keep more steps for clearer directions."""
    if not polyline:
        return []
    pts = _sample_points(polyline, max_pts=max_pts)
    if not pts:
        return []
    def haversine(a, b):
        lat1, lon1 = a; lat2, lon2 = b
        R = 6371000.0
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        sa = math.sin(dlat/2.0); sb = math.sin(dlon/2.0)
        aa = sa*sa + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*sb*sb
        return 2*R*math.atan2(math.sqrt(aa), math.sqrt(1-aa))
    tot_len = sum(haversine(pts[i-1], pts[i]) for i in range(1, len(pts)))
    raw = []
    last = None
    for (lat, lon) in pts:
        nm = reverse_cache_lookup(lat, lon) or None
        if not raw or (nm or "") != (last or ""):
            raw.append({"name": nm or "Unnamed", "coords": [(lat, lon)]})
            last = nm or "Unnamed"
        else:
            raw[-1]["coords"].append((lat, lon))
    MIN_SEG_LEN = 40.0
    MIN_SHARE = 0.015
    strict = []
    for idx, seg in enumerate(raw):
        nm = (seg["name"] or "").strip()
        if not nm or nm.lower() == "unnamed":
            continue
        coords = seg.get("coords") or []
        if len(coords) < 2:
            continue
        length = sum(haversine(coords[i-1], coords[i]) for i in range(1, len(coords)))
        share = (length/tot_len) if tot_len>0 else 0.0
        if idx in (0, len(raw)-1) or length >= MIN_SEG_LEN or share >= MIN_SHARE:
            strict.append({"name": nm, "coords": coords})
    merged = []
    for seg in strict:
        if not merged or merged[-1]["name"].lower() != seg["name"].lower():
            merged.append({"name": seg["name"], "coords": list(seg["coords"])})
        else:
            merged[-1]["coords"].extend(seg["coords"])
    return merged

def describe_turns_sentence(route_dict: dict, label: str | None = None, *, max_segments: int = 14):
    """Prefer map-matching steps (maneuvers) for Maps-like directions; fallback to on-route segments."""
    season = (route_dict.get("Season") or route_dict.get("season") or "").strip().lower()
    segs = []
    if season == "road":
        try:
            segs = radar_match_steps(route_dict.get("polyline"))
        except Exception:
            segs = []
    if not segs:
        season = (route_dict.get("Season") or route_dict.get("season") or "").strip().lower()
    segs = []
    prelist_for_must = []
    if season == "road":
        try:
            segs = radar_match_steps(route_dict.get("polyline"), mode="foot", unique=True)
            # also get a rich prelist for Must Roads checks
            prelist_for_must = radar_match_steps(route_dict.get("polyline"), mode="foot", unique=False)
        except Exception:
            segs = []
            prelist_for_must = []
    if not segs:
        steps = locationiq_match_steps(route_dict.get("polyline"))
        if steps:
            segs = steps
            prelist_for_must = steps
        else:
            segs = onroute_named_segments(route_dict.get("polyline"))
            prelist_for_must = list(segs)
    if not segs:
        return ""

    # Dynamic segment budget & anti-truncation for short routes (5k)
    if label:
        lab = label.strip().lower()
        if lab.startswith("5"):
            max_segments = max(max_segments, 26)
        elif lab.startswith("8"):
            max_segments = max(max_segments, 24)

    # If road route produced too few names, try non-unique steps then condense
    if (route_dict.get("Season","").lower() == "road") and len(segs) <= 3:
        try:
            raw_steps = radar_match_steps(route_dict.get("polyline"), mode="foot", unique=False) or []
            # compress consecutive duplicates by canonical name
            def _canon(n): 
                import re
                s = re.sub(r"^[A-Z]{1,2}\d+\s+","",(n or "").strip())
                s = re.sub(r"\brd\b|\brd\.\b","Road",s,flags=re.I)
                s = re.sub(r"\bln\b|\bln\.\b","Lane",s,flags=re.I)
                s = re.sub(r"\bave\b|\bave\.\b","Avenue",s,flags=re.I)
                return re.sub(r"\s+"," ",s).strip()
            condensed = []
            prevk = None
            for s in raw_steps:
                k = _canon(s.get("name") or "").lower()
                if not k: 
                    continue
                if k != prevk:
                    condensed.append({"name": _canon(s.get("name")), "coords": s.get("coords")})
                    prevk = k
            # unique-first preserving order
            seen=set(); uniq=[]
            for s in condensed:
                k = _canon(s.get("name") or "").lower()
                if k and k not in seen:
                    seen.add(k); uniq.append(s)
            if len(uniq) > len(segs):
                segs = uniq
        except Exception:
            pass

    # Force-include from sheet based on label (8k/5k)
    must_csv = ""
    if label and label.strip().lower().startswith("8"):
        must_csv = route_dict.get("Must Roads A") or route_dict.get("Must Roads 1") or route_dict.get("Must Roads (8k)") or ""
    elif label and label.strip().lower().startswith("5"):
        must_csv = route_dict.get("Must Roads B") or route_dict.get("Must Roads 2") or route_dict.get("Must Roads (5k)") or ""
    segs = must_include_from_sheet(segs, segs, must_csv)

    # Optional elevation slope for first segment (if GPX available)
    elev_pts = None
    try:
        rid = _extract_route_id(route_dict.get("url") or route_dict.get("rid"))
        tok = get_strava_token()
        if rid and tok:
            elev_pts = fetch_strava_route_gpx_points(rid, tok)
    except Exception:
        pass

    def _turn_from_modifier(man):
        mod = (man or {}).get("modifier")
        if not mod:
            return "continue onto"
        m = str(mod).lower()
        if "uturn" in m: return "make a U-turn onto"
        if "slight" in m and "right" in m: return "bear right onto"
        if "slight" in m and "left"  in m: return "bear left onto"
        if m == "right": return "turn right onto"
        if m == "left":  return "turn left onto"
        if "straight" in m: return "continue onto"
        return "continue onto"

    def _nearest_el(lat, lon):
        if not elev_pts: return None
        best, bestd = None, 1e9
        for (la, lo, el) in elev_pts:
            d = (la-lat)*(la-lat)+(lo-lon)*(lo-lon)
            if d < bestd and el is not None:
                bestd, best = d, el
        return best

    def _segment_slope(coords):
        if not coords or len(coords) < 2 or not elev_pts:
            return 0.0
        s = _nearest_el(*coords[0]); e = _nearest_el(*coords[-1])
        if s is None or e is None: return 0.0
        return float(e - s)

    path_like = ("path","towpath","trail","canal","promenade","greenway","footpath")

    parts = []
    # First segment
    first = segs[0]
    delta_el = _segment_slope(first.get("coords"))
    verb0 = "up" if delta_el > 3 else ("down" if delta_el < -3 else "along")
    name0 = first["name"]
    art0 = "the " if any(t in name0.lower() for t in path_like) and not name0.lower().startswith("the ") else ""
    parts.append(f"{verb0} {art0}{name0}".strip())

    # Subsequent segments
    for i in range(1, min(len(segs), max_segments)):
        cur = segs[i]
        nm = cur["name"]
        art = "the " if any(t in nm.lower() for t in path_like) and not nm.lower().startswith("the ") else ""
        if "maneuver" in cur:
            connector = _turn_from_modifier(cur.get("maneuver"))
        else:
            connector = "then onto"
        parts.append(f"{connector} {art}{nm}")

    return "We’ll be running " + ", ".join(parts) + "."

def route_blurb(label, r: dict) -> str:
    if isinstance(r.get("dist"), (int, float)):
        dist_txt = f"{r['dist']:.1f} km"
    elif r.get("dist") is not None:
        dist_txt = f"{r['dist']} km"
    else:
        dist_txt = "? km"
    url = r.get("url") or (f"https://www.strava.com/routes/{r.get('rid')}" if r.get("rid") else "")
    name = r.get("name") or "Route"
    line1 = f"• {label} – {name}" + (f": {url}" if url else "")
    elev_part = ""
    if isinstance(r.get("elev"), (int, float)):
        elev_part = f" with {r['elev']:.0f}m of elevation"
    line2 = f"  {dist_txt}{elev_part}"
    return "\n".join([line1, line2])

def sort_with_labels(r1, r2):
    def d(r): return r["dist"] if r["dist"] is not None else -1
    a, b = (r1, r2) if d(r1) >= d(r2) else (r2, r1)
    return [("8k", a), ("5k", b)]

labeled = sort_with_labels(routes[0], routes[1])

# Deterministic random seed based on date so text varies week-to-week but stays stable per date
date_key = str(row.get("_dateonly") or "")
seed = int(hashlib.sha1(date_key.encode("utf-8")).hexdigest()[:8], 16) if date_key else random.randint(0, 2**32 - 1)
rng_email = random.Random(seed)
rng_fb = random.Random(seed + 1)
rng_wa = random.Random(seed + 2)

# Common intro variants
intro_variants = [
    "We’ve got {num_routes} routes lined up and {num_options} great options this week:",
    "This Thursday we’ve got {num_routes} routes planned and {num_options} great options to choose from:",
    "{num_routes} routes, {num_options} great options – something for everyone this Thursday:",
    "Fancy a run or walk? We’ve planned {num_routes} routes and {num_options} options for you this Thursday:",
    "Whether you’re chasing a PB or fancy a chatty one, we’ve got {num_routes} routes and {num_options} options to pick from:",
]

closing_variants_email = [
    "Grab your spot and come run/walk with us! 🧡",
    "Fancy joining us this week? Book your spot and come along! 🧡",
    "We’d love to see you there – grab a place and join the fun! 🧡",
]

closing_variants_fb = [
    "Tag a friend who might like to join us and share the running love! 🧡",
    "Know someone who’d enjoy this? Tag them and bring them along! 🧡",
    "New faces always welcome – tag a friend and spread the word! 🧡",
]

closing_variants_wa = [
    "*We set off at 7:00pm – please book on and arrive a few minutes early.*",
    "*We set off at 7:00pm – book your spot and come a little early to say hi.*",
    "*We set off at 7:00pm – grab a place and aim to arrive a few minutes before.*",
]

def build_intro_line(rng: random.Random, include_jeffing: bool) -> str:
    """Build an intro line that reflects the actual number of routes/options."""
    # Base routes: 5k and 8k
    num_routes = 2 + (1 if route3 is not None else 0)
    # Options: 5k + 8k + optional Jeffing + optional walk/C25K (Route 3)
    num_options = 2 + (1 if include_jeffing else 0) + (1 if route3 is not None else 0)
    tmpl = intro_variants[rng.randint(0, len(intro_variants) - 1)]
    return tmpl.format(num_routes=num_routes, num_options=num_options)
def build_route_option_lines(include_jeffing: bool) -> list[str]:
    """Build the list of session options for the intro line.

    Only include the walk/C25K option if Route 3 is actually defined
    (i.e. there is a non-empty description). Jeffing, 5k and 8k are
    then added as usual.
    """
    opts: list[str] = []

    has_route3 = route3 is not None and bool((route3_desc or "").strip())
    if has_route3:
        label3 = (route3_desc or "Walk").strip() or "Walk"
        emoji3 = "🚶" if "walk" in label3.lower() else "🏃"
        opts.append(f"{emoji3} {label3}")

    if include_jeffing:
        opts.append("🏃 Jeffing")

    # 5k and 8k options are always available
    opts.append("🏃 5k")
    opts.append("🏃‍♀️ 8k")

    return opts

def build_common_meeting_lines(include_map: bool = True) -> list[str]:
    lines: list[str] = []
    if is_on_tour_meeting:
        lines.append(f"📍 This week we’re On Tour – meeting at {nice_meet_loc} at 7pm")
        if include_map and meet_map_url:
            lines.append(f"🗺️ Meeting point map: {meet_map_url}")
    else:
        lines.append(f"📍 Meeting at: {nice_meet_loc} at 7pm")
    return lines

def build_route_detail_lines() -> list[str]:
    lines: list[str] = []
    lines.append("This week’s routes")
    lines.append("")
    # Route 3 (walk/C25K) first if present
    label3 = (route3_desc or "Walk").strip() or "Walk"
    if route3 is not None:
        lines.append(route_blurb(label3, route3))
        lines.append("")
    # Then 8k and 5k from labeled list
    for label, r in labeled:
        lines.append(route_blurb(label, r))
        lines.append("")
    # Trim trailing blank lines
    while lines and not lines[-1].strip():
        lines.pop()
    lines.append("")
    lines.append("📍 If you want to look ahead, our upcoming schedule is available at this link:")
    lines.append("https://runtogetherradcliffe.github.io/weeklyschedule")
    return lines

def build_safety_and_weather_lines() -> list[str]:
    """Combine safety and weather guidance into a single friendly paragraph where possible."""
    safety = SAFETY_NOTE if is_road else ""
    weather_line = get_weather_blurb_for_date(row.get("_dateonly"))
    if safety and weather_line:
        # Lowercase first letter of weather line so it flows after "Also"
        wl = weather_line
        if wl and wl[0].isupper():
            wl = wl[0].lower() + wl[1:]
        return [f"{safety} Also {wl}"]
    elif safety:
        return [safety]
    elif weather_line:
        return [weather_line]
    return []

BOOKING_BLOCK = [
    "📲 The easiest way to manage your bookings is via the RunTogether Runner app on your phone:",
    "· Download the Apple iOS app: https://apps.apple.com/gb/app/runtogether-runner/id1447488812",
    "· Download the Android app: https://play.google.com/store/apps/details?id=com.sportlabs.android.runner&pcampaignid=web_share",
    "",
    "💻 Prefer the website? You can also book via:",
    "· https://clubspark.englandathletics.org/RunTogetherRadcliffe/Coaching",
    "To cancel you need to use this link:",
    "· https://clubspark.englandathletics.org/EnglandAthleticsRunTogether/Courses",
    "",
    "ℹ️ Cancellation info:",
    "· On the website you can cancel until midnight the day before.",
    "· On the app you can cancel up to 1 hour before the session.",
]

def build_email_booking_block() -> list[str]:
    """Email-specific booking block for the plain-text email version."""
    lines: list[str] = []
    lines.append("📲 The easiest way to manage your bookings is via the RunTogether Runner app on your phone:")
    lines.append("· Download the Apple iOS app here: https://apps.apple.com/gb/app/runtogether-runner/id1447488812")
    lines.append("· Download the Android app here: https://play.google.com/store/apps/details?id=com.sportlabs.android.runner&pcampaignid=web_share")
    lines.append("")
    lines.append("💻 Prefer the website? You can also book via:")
    lines.append("· https://clubspark.englandathletics.org/RunTogetherRadcliffe/Coaching")
    lines.append("To cancel you need to use this link:")
    lines.append("· https://clubspark.englandathletics.org/EnglandAthleticsRunTogether/Courses")
    lines.append("")
    lines.append("ℹ️ Cancellation info:")
    lines.append("· On the website you can cancel until midnight the day before.")
    lines.append("· On the app you can cancel up to 1 hour before the session.")
    return lines

# ----------------------------
# Email text
# ----------------------------
email_lines: list[str] = []
email_lines.append(build_intro_line(rng_email, show_jeffing))
email_lines.extend(build_route_option_lines(show_jeffing))
email_lines.append("")
email_lines.extend(build_common_meeting_lines(include_map=True))
email_lines.append("")
detail_lines = build_route_detail_lines()
email_lines.extend(detail_lines)
email_lines.append("")
email_lines.append("How to book")
email_lines.append("")
email_lines.extend(build_email_booking_block())
email_lines.append("")
email_lines.append("Additional information")
email_lines.append("")
email_lines.extend(build_safety_and_weather_lines())
email_lines.append("")
email_lines.append(closing_variants_email[rng_email.randint(0, len(closing_variants_email) - 1)])
email_text = "\n".join(email_lines)

def make_email_html(email_text: str) -> str:
    """Convert the plain email text into HTML with bold section headers and embedded app links.
    This is used to render a rich preview that can be copy–pasted into an email editor.
    """
    import html as _html
    import re as _re

    lines = email_text.split("\n")
    html_lines: list[str] = []

    for line in lines:
        stripped = line.strip()

        # Bold section headings
        if stripped in {"This week’s routes", "This week's routes"}:
            html_lines.append("<b>This week’s routes</b>")
            continue
        if stripped == "How to book":
            html_lines.append("<b>How to book</b>")
            continue
        if stripped == "Additional information":
            html_lines.append("<b>Additional information</b>")
            continue

        # App download lines: embed links on "here"
        m_ios = _re.match(r"(.*Download the Apple iOS app) here: (https?://\S+)", stripped)
        if m_ios:
            prefix, url = m_ios.group(1), m_ios.group(2)
            html_lines.append(
                f"{_html.escape(prefix)} <a href=\"{_html.escape(url, quote=True)}\">here</a>"
            )
            continue

        m_and = _re.match(r"(.*Download the Android app) here: (https?://\S+)", stripped)
        if m_and:
            prefix, url = m_and.group(1), m_and.group(2)
            html_lines.append(
                f"{_html.escape(prefix)} <a href=\"{_html.escape(url, quote=True)}\">here</a>"
            )
            continue

        # Default: escape as normal text (strip leading spaces to avoid odd indentation)
        html_lines.append(_html.escape(line.lstrip()))

    # Join with <br> so line breaks are preserved
    return "<br>".join(html_lines)

# ----------------------------
# Facebook post
# ----------------------------
fb_lines: list[str] = []
fb_lines.append(f"RunTogether Radcliffe – this Thursday {date_str}")
fb_lines.append("")
fb_lines.append(build_intro_line(rng_fb, show_jeffing))
fb_lines.extend(build_route_option_lines(show_jeffing))
fb_lines.append("")
fb_lines.extend(build_common_meeting_lines(include_map=True))
fb_lines.append("")
fb_lines.extend(BOOKING_BLOCK)
fb_lines.extend(build_route_detail_lines())
fb_lines.append("")
fb_lines.extend(build_safety_and_weather_lines())
fb_lines.append("")
fb_lines.append(closing_variants_fb[rng_fb.randint(0, len(closing_variants_fb) - 1)])
facebook_text = "\n".join(fb_lines)

# ----------------------------
# WhatsApp message
# ----------------------------
wa_lines: list[str] = []
wa_lines.append(f"*RunTogether Radcliffe – Thursday {date_str}*")
wa_lines.append("")
wa_lines.append(build_intro_line(rng_wa, show_jeffing))
for opt in build_route_option_lines(show_jeffing):
    wa_lines.append(f"- {opt}")
wa_lines.append("")
for line in build_common_meeting_lines(include_map=True):
    wa_lines.append(line)
wa_lines.append("")
wa_lines.append("Route links for this week (and future runs):")
wa_lines.append("https://runtogetherradcliffe.github.io/weeklyschedule")
wa_lines.append("")
wa_lines.extend(build_safety_and_weather_lines())
wa_lines.append("")
wa_lines.append(closing_variants_wa[rng_wa.randint(0, len(closing_variants_wa) - 1)])
wa_text = "\n".join(wa_lines)

st.subheader("Email (copy from here)")
email_html = make_email_html(email_text)
components.html(f"""
<div>
  <button onclick="(function() {{
      try {{
          const node = document.getElementById('email-html');
          if (!node) return;
          const range = document.createRange();
          range.selectNodeContents(node);
          const sel = window.getSelection();
          sel.removeAllRanges();
          sel.addRange(range);
          document.execCommand('copy');
      }} catch (e) {{
          console.error(e);
      }}
  }})()" style="margin-bottom:0.5rem;">
    Copy email
  </button>
  <div id="email-html" style="white-space:pre-wrap; font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;">
    {email_html}
  </div>
</div>
""", height=900)

st.download_button(
    "Download email as HTML",
    data=email_html,
    file_name="rtr_thursday_email.html",
    mime="text/html",
)

st.subheader("Facebook post")
st.text_area("Facebook post", value=facebook_text, height=320)

st.subheader("WhatsApp message")
st.text_area("WhatsApp message", value=wa_text, height=320)
