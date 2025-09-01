
# pages/social_posts.py
# Build: v2025.09.01-SOCIAL-24 (rebuild: robust date-only + Strava/LocationIQ enrichment)

import re
import random
import urllib.parse
from datetime import datetime
import pandas as pd
import requests
import hashlib
import streamlit as st
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
    return (str(s).strip()) if s is not None else ""

def try_float(x):
    try:
        if x is None or x == "":
            return None
        return float(x)
    except Exception:
        return None

def make_https(url):
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




def locationiq_pois(polyline=None, sample_points=None, *, rid: str | None = None, debug: bool = False):
    """
    Return up to 3 highlights drawn from features along the route.
    - Caches reverse geocoding results (6h) and final POIs per route/polyline hash.
    - If debug=True, returns (pois, report) where report describes the pipeline.
    """
    import time

    base_pref = _get_secret("LOCATIONIQ_BASE") or "us1"
    base_order = []
    for b in (base_pref, "us1", "eu1", "ap1"):
        if b not in base_order:
            base_order.append(b)

    key = _get_secret("LOCATIONIQ_API_KEY") or _get_secret("LOCATIONIQ_TOKEN")
    if not key:
        return ([], {"error": "no_api_key"}) if debug else []

    pts = sample_points or _sample_points(polyline, max_pts=30)
    if not pts:
        return ([], {"error": "no_points"}) if debug else []

    ONROUTE_KEYS = [
        "road","footway","pedestrian","path","cycleway","bridleway","trail","steps",
        "bridge","river","waterway","park","leisure","natural"
    ]
    PRIORITY_WORDS = ["trail","canal","river","park","bridge","viaduct","towpath","nature","reserve"]
    ROAD_WORDS = ["road","lane","street","way","drive","avenue","rd","ln","st"]

    # Cache keys (per final POI result)
    phash = hashlib.md5((polyline or str(pts)).encode("utf-8")).hexdigest()
    cache_key = f"pois:{rid or ''}:{phash}:{base_order[0]}"
    if "poi_cache" not in st.session_state:
        st.session_state["poi_cache"] = {}
    if not debug:
        cached = st.session_state["poi_cache"].get(cache_key)
        if cached:
            return cached

    # Reverse geocode with caching of each call
    if hasattr(st, "cache_data"):
        @st.cache_data(show_spinner=False, ttl=21600)
        def _liq_reverse_cached(base: str, key: str, lat: float, lon: float, zoom: int):
            try:
                r = requests.get(
                    f"https://{base}.locationiq.com/v1/reverse",
                    params={
                        "key": key,
                        "lat": f"{lat:.6f}",
                        "lon": f"{lon:.6f}",
                        "format": "json",
                        "normalizeaddress": 1,
                        "addressdetails": 1,
                        "namedetails": 1,
                        "zoom": zoom,
                    },
                    timeout=12,
                )
                if not r.ok:
                    return None
                return r.json()
            except Exception:
                return None
    else:
        def _liq_reverse_cached(base: str, key: str, lat: float, lon: float, zoom: int):
            try:
                r = requests.get(
                    f"https://{base}.locationiq.com/v1/reverse",
                    params={
                        "key": key,
                        "lat": f"{lat:.6f}",
                        "lon": f"{lon:.6f}",
                        "format": "json",
                        "normalizeaddress": 1,
                        "addressdetails": 1,
                        "namedetails": 1,
                        "zoom": zoom,
                    },
                    timeout=12,
                )
                if not r.ok:
                    return None
                return r.json()
            except Exception:
                return None

    def extract_names(payload: dict) -> list[str]:
        out = []
        addr = payload.get("address") or {}
        namedetails = payload.get("namedetails") or {}
        disp = payload.get("display_name") or ""
        for k in ONROUTE_KEYS:
            v = addr.get(k)
            if v and v.strip() and v.lower() != "unnamed road":
                out.append(v.strip())
        nm = namedetails.get("name")
        if nm and nm.strip():
            out.append(nm.strip())
        if disp:
            first = disp.split(",")[0].strip()
            if first and first.lower() != "unnamed road":
                out.append(first)
        return out

    names = []
    hits = 0
    calls = 0
    per_point: list[dict] = []
    for (lat, lon) in pts:
        point_report = {"lat": lat, "lon": lon, "tried": [], "got": None}
        got = False
        for b in base_order:
            for z in (18, 16):
                calls += 1
                payload = _liq_reverse_cached(b, key, lat, lon, z)
                point_report["tried"].append({"base": b, "zoom": z, "ok": payload is not None})
                if payload:
                    cands = extract_names(payload)
                    if cands:
                        names.extend(cands)
                    point_report["got"] = (cands[:3] if cands else [])
                    hits += 1
                    got = True
                    break
            if got:
                break
        per_point.append(point_report)
        if hits >= 12:  # enough per route
            break

    # Filtering & ranking
    clean = []
    for n in names:
        s = n.strip()
        if not s or s.lower() == "unnamed road":
            continue
        l = s.lower()
        ok = any(w in l for w in PRIORITY_WORDS) or any(w in l for w in ROAD_WORDS) or (len(s.split()) >= 2 and s[0].isupper())
        if ok:
            clean.append(s)

    seen=set(); dedup=[]
    for s in clean:
        k=s.lower()
        if k in seen: 
            continue
        seen.add(k); dedup.append(s)

    def score(s: str) -> int:
        l = s.lower()
        if any(w in l for w in ["trail","canal","river","park","bridge","viaduct","towpath","nature","reserve"]):
            return 0
        return 1

    dedup.sort(key=score)
    pois = dedup[:3]

    # store cache
    st.session_state["poi_cache"][cache_key] = pois

    if debug:
        report = {
            "rid": rid,
            "polyline_hash": phash,
            "points_considered": len(pts),
            "reverse_calls": calls,
            "points_hit": hits,
            "raw_names": names[:30],
            "filtered": dedup[:10],
            "final_pois": pois,
            "bases": base_order,
        }
        return pois, report
    return pois

    pts = sample_points or _sample_points(polyline, max_pts=18)  # slightly denser
    if not pts:
        return []

    WANT_KEYS = [
        "road","footway","pedestrian","path","cycleway","bridleway",
        "trail","steps","bridge"
    ]
    # POI-ish feature classes and types we'll accept by name
    PRIORITY_WORDS = ["trail","canal","river","park","bridge","viaduct","nature","reserve","reservoir","country park"]
    ROAD_WORDS = ["road","lane","street","way","drive","avenue","rd","ln","st"]

    def pick_name(payload: dict) -> list[str]:
        out = []
        addr = payload.get("address") or {}
        namedetails = payload.get("namedetails") or {}
        disp = payload.get("display_name") or ""

        # 1) Named road/path along the coordinate
        for k in WANT_KEYS:
            val = addr.get(k)
            if val and val.strip() and val.lower() != "unnamed road":
                out.append(val.strip())

        # 2) Named feature (namedetails.name)
        nm = namedetails.get("name")
        if nm and nm.strip():
            out.append(nm.strip())

        # 3) First part of display_name as a fallback
        if disp:
            first = disp.split(",")[0].strip()
            if first and first.lower() != "unnamed road":
                out.append(first)

        return out

    names: list[str] = []
    # Reverse geocode with light throttling to avoid free-tier rate limits
    for (lat, lon) in pts:
        for b in bases:
            try:
                resp = requests.get(
                    f"https://{b}.locationiq.com/v1/reverse",
                    params={
                        "key": key,
                        "lat": f"{lat:.6f}",
                        "lon": f"{lon:.6f}",
                        "format": "json",
                        "normalizeaddress": 1,
                        "addressdetails": 1,
                        "namedetails": 1,
                        "zoom": 18,
                    },
                    timeout=12,
                )
                if resp.ok:
                    payload = resp.json()
                    cand = pick_name(payload)
                    # extend; we'll filter later
                    names.extend(cand)
                    break
            except Exception:
                continue
        # throttle ~2 req/s across bases
        time.sleep(0.45)

        # stop early if we've already gathered plenty
        if len(names) > 30:
            break

    # Filter: only keep items that look like on-route roads/POIs
    clean = []
    for n in names:
        s = n.strip()
        if not s: 
            continue
        low = s.lower()
        if low in ("unnamed road",):
            continue
        # Cheap heuristics: accept if it contains priority words or common road words,
        # and reject building numbers or single tokens that look like postcodes.
        ok = any(w in low for w in PRIORITY_WORDS) or any(w in low for w in ROAD_WORDS)
        if not ok:
            # also accept multi-word proper names (e.g., "Outwood Trail")
            ok = (len(s.split()) >= 2 and s[0].isupper())
        if ok:
            clean.append(s)

    # Ordered de-dup, but collapse consecutive duplicates
    dedup = []
    seen = set()
    prev = None
    for n in clean:
        k = n.lower()
        if k == prev:
            continue
        prev = k
        if k not in seen:
            seen.add(k)
            dedup.append(n)

    # Prefer POI-flavoured names first, then roads
    def score(n: str) -> int:
        l = n.lower()
        if any(w in l for w in ["canal","river","trail","park","bridge","viaduct","nature","reserve"]):
            return 0
        return 1

    dedup.sort(key=score)
    return dedup[:3]

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

sheet_url = st.text_input("Google Sheet URL", value=get_cfg("SHEET_CSV_URL", get_cfg("SHEET_URL","")))
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

with st.expander("Debug: POIs (per-route)", expanded=False):
    try:
        st.json(poi_debug if poi_debug else [{"note":"no debug collected"}])
    except Exception:
        st.write(poi_debug if poi_debug else [{"note":"no debug collected"}])
poi_debug = []
try:
    for r in routes:
        if r.get('rid'):
            _, rep = locationiq_pois(None, sample_points=_sample_points(r.get('polyline')) if r.get('polyline') else None, rid=r.get('rid'), debug=True)
            poi_debug.append(rep if isinstance(rep, dict) else {'rid': r.get('rid'), 'note':'no_rep'})
except Exception:
    pass


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

date_str = format_day_month_uk(pd.to_datetime(row["_dateonly"]))
time_line = "🕖 We set off at 7:00pm"
meeting_line = f"📍 Meeting at: {meet_loc.title()}"

def route_blurb(label, r: dict) -> str:
    if isinstance(r.get("dist"), (int,float)):
        dist_txt = f"{r['dist']:.1f} km"
    elif r.get("dist") is not None:
        dist_txt = f"{r['dist']} km"
    else:
        dist_txt = "? km"
    desc = hilliness_blurb(r.get("dist"), r.get("elev"))
    url = r.get("url") or (f"https://www.strava.com/routes/{r.get('rid')}" if r.get("rid") else "")
    name = r.get("name") or "Route"
    line1 = f"• {label} – {name}" + (f": {url}" if url else "")
    elev_part = f" with {r['elev']:.0f}m of elevation" if isinstance(r.get("elev"), (int,float)) else ""
    line2 = f"  {dist_txt}{elev_part} – {desc}"
    highlights = ""
    if r.get("pois"):
        parts = []
        for ch in str(r["pois"]).split("|"):
            parts.extend([p.strip() for p in re.split(r"[;,]", ch) if p.strip()])
        if parts:
            # dedupe and keep first 3
            seen=set(); uniq=[]
            for p in parts:
                k=p.lower()
                if k not in seen:
                    seen.add(k); uniq.append(p)
            highlights = "🏞️ Highlights: " + ", ".join(uniq[:3])
    lines = [line1, line2]
    if highlights: lines.append("  " + highlights)
    return "\n".join(lines)

# Order long/short by distance if available
def sort_with_labels(r1, r2):
    def d(r): return r["dist"] if r["dist"] is not None else -1
    a, b = (r1, r2) if d(r1) >= d(r2) else (r2, r1)
    return [("8k", a), ("5k", b)]

labeled = sort_with_labels(routes[0], routes[1])

lines = []
lines.append(f"🏃 This Thursday — {date_str}")
lines.append("")
lines.append(meeting_line)
lines.append(time_line)
lines.append("")
lines.append("🛣️ This week we’ve got two route options to choose from:")
lines.append(route_blurb(labeled[0][0], labeled[0][1]))
lines.append(route_blurb(labeled[1][0], labeled[1][1]))
lines.append("")
lines.append("📲 Book now:")
lines.append("https://groups.runtogether.co.uk/RunTogetherRadcliffe/Runs")
lines.append("❌ Can’t make it? Cancel at least 1 hour before:")
lines.append("https://groups.runtogether.co.uk/My/BookedRuns")
lines.append("")
lines.append("👟 Grab your shoes, bring your smiles – see you Thursday!")
lines.append("")
lines.append("*RunTogether Radcliffe – This Thursday!*")

post_text = "\n".join(lines)

st.subheader("Composed message")
st.text_area("Copy/paste to socials", value=post_text, height=420)
