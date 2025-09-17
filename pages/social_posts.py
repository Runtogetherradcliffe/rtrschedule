import streamlit as st

# pages/social_posts.py
# Build: v2025.09.01-SOCIAL-24 (rebuild: robust date-only + Strava/LocationIQ enrichment)

import re
import random
import urllib.parse
from datetime import datetime





@st.cache_data(ttl=7*24*3600, show_spinner=False)
def cached_onroute_segments(polyline: str, max_pts: int = 240):
    try:
        return onroute_named_segments(polyline, max_pts=max_pts)
    except Exception:
        return []

# --- Elevation-assisted phrasing (up/down) ---
import requests
from math import radians, sin, cos, atan2, sqrt

def _hv_dist(a, b):
    # meters
    lat1, lon1 = a; lat2, lon2 = b
    R = 6371000.0
    dlat = radians(lat2 - lat1); dlon = radians(lon2 - lon1)
    sa = sin(dlat/2.0); sb = sin(dlon/2.0)
    aa = sa*sa + cos(radians(lat1))*cos(radians(lat2))*sb*sb
    return 2*R*atan2(sqrt(aa), sqrt(1-aa))

@st.cache_data(ttl=14*24*3600, show_spinner=False)
def _elevations_for_points(points: list[tuple]) -> list | None:
    """Fetch elevations (meters) for a list of (lat,lon) using open-elevation; returns list aligned to input or None on failure."""
    if not points:
        return []
    try:
        # Batch up to ~90 per request
        elevs = []
        batch = 90
        for i in range(0, len(points), batch):
            chunk = points[i:i+batch]
            locs = "|".join(f"{lat:.6f},{lon:.6f}" for (lat,lon) in chunk)
            url = f"https://api.open-elevation.com/api/v1/lookup?locations={locs}"
            resp = requests.get(url, timeout=10)
            if resp.status_code != 200:
                return None
            data = resp.json().get("results", [])
            if len(data) != len(chunk):
                return None
            elevs.extend([r.get("elevation") for r in data])
        return elevs
    except Exception:
        return None

def _segment_grade(seg_coords: list[tuple], elev_map: dict) -> float:
    """Return average grade (rise/run) for a segment using first/last points; positive = uphill."""
    if not seg_coords:
        return 0.0
    a = seg_coords[0]; b = seg_coords[-1]
    e1 = elev_map.get(a); e2 = elev_map.get(b)
    if e1 is None or e2 is None:
        return 0.0
    run = max(_hv_dist(a,b), 1.0)
    return (e2 - e1) / run

def _apply_updown(sentence: str, segs: list[dict], grade_by_index: list[float], thr: float = 0.012, outback_names: set[str] | None = None) -> str:
    """Replace 'along/onto {name}' with 'up/down' where grade magnitude exceeds threshold. One-pass, ordered replacements."""
    s = sentence
    for idx, seg in enumerate(segs):
        if outback_names and (seg.get('name') or '').strip().lower() in outback_names:
            # Neutral phrasing for out-and-back roads
            continue
        nm = (seg.get("name") or "").strip()
        if not nm: 
            continue
        g = grade_by_index[idx] if idx < len(grade_by_index) else 0.0
        if abs(g) < thr:
            continue
        # Prefer replacing 'onto {name}' first, else 'along {name}' (first occurrence only)
        if g > 0:
            s_new = s.replace(f"onto {nm}", f"up {nm}", 1)
            if s_new == s:
                s_new = s.replace(f"along {nm}", f"up {nm}", 1)
        else:
            s_new = s.replace(f"onto {nm}", f"down {nm}", 1)
            if s_new == s:
                s_new = s.replace(f"along {nm}", f"down {nm}", 1)
        s = s_new
    return s

# --- Fair, interleaved reverse-geocode prefetch to avoid starving the 2nd route ---
def _prefetch_reverse_for_routes(routes: list[dict], max_pts_each: int = 180) -> int:
    try:
        # Build per-route sampled points
        samples = []
        for r in routes:
            pl = r.get("polyline")
            pts = _sample_points(pl, max_pts=max_pts_each) if pl else []
            samples.append(pts)
        # Interleave calls: round-robin through the lists
        calls = 0
        idx = 0
        more = True
        while more:
            more = False
            for pts in samples:
                if idx < len(pts):
                    lat, lon = pts[idx]
                    try:
                        # Reverse lookup is usually cached; this will be a no-op if already fetched
                        _ = reverse_cache_lookup(lat, lon)
                        calls += 1
                    except Exception:
                        pass
                    more = True
            idx += 1
        return calls
    except Exception:
        return 0
# --- Cached directions helpers (single source of truth) ---
@st.cache_data(ttl=7*24*3600, show_spinner=False)
def _cached_directions_sentence(polyline: str | None, url_or_rid: str | None, max_segments: int = 26) -> str:
    """Cached wrapper around describe_turns_sentence; keyed by polyline+route id/url."""
    try:
        rd = {"polyline": polyline, "url": url_or_rid, "rid": url_or_rid}
        return describe_turns_sentence(rd, max_segments=max_segments)
    except TypeError:
        # Back-compat if describe_turns_sentence doesn't accept max_segments
        try:
            return describe_turns_sentence(rd)
        except Exception:
            return ""
    except Exception:
        return ""

def cached_sentence_for_route(r: dict, max_segments: int = 26) -> str:
    return _cached_directions_sentence(r.get("polyline"), r.get("url") or r.get("rid"), max_segments)

SAFETY_NOTE = "As we are now running after dark, please remember lights and hi-viz, be safe, be seen!"
import pandas as pd
import requests
import os
import hashlib

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
    Fast POI finder for social post (v24.15):
      - Budgeted runtime (~2.5s)
      - Sample up to 24 points along line; reverse at zooms 17,16
      - Prioritise towpath/canal/woods/viaduct/bridge/river and later-route features
      - Penalise generic starts like "Outwood Gate"/"Outwood"/supermarket names
    """
    import time

    t0 = time.perf_counter()
    BUDGET = 2.5  # seconds soft budget

    base_pref = _get_secret("LOCATIONIQ_BASE") or "us1"
    base_order = []
    for b in (base_pref, "us1", "eu1", "ap1"):
        if b not in base_order:
            base_order.append(b)

    key, key_src = get_locationiq_key(debug=True)
    if not key:
        return ([], {"error": "no_api_key", "api_key_source": key_src}) if debug else []

    pts = sample_points or _sample_points(polyline, max_pts=24)
    if not pts:
        return ([], {"error": "no_points", "api_key_source": key_src}) if debug else []

    ONROUTE_KEYS = [
        "road","footway","pedestrian","path","cycleway","bridleway","trail","steps",
        "bridge","river","waterway","canal","park","leisure","natural"
    ]
    PRIORITY_WORDS = ["towpath","canal","river","wood","woods","forest","viaduct","bridge","trail","park","nature","reserve"]
    ROAD_WORDS = ["road","lane","street","way","drive","avenue","rd","ln","st"]
    BANLIST = {s.lower() for s in [
        "outwood gate","outwood","lidl","radcliffe market","pilkington way","dale street","bank field street",
        "asda pay desk","radcliffe bus station","phoenix way","sion street","highmeadow"
    ]}

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
                        "extratags": 1,
                        "zoom": zoom,
                    },
                    timeout=8,
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
                        "extratags": 1,
                        "zoom": zoom,
                    },
                    timeout=8,
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
        klass = (payload.get("class") or "").lower()
        typ = (payload.get("type") or "").lower()
        xtra = payload.get("extratags") or {}

        if klass in ("waterway","natural","leisure"):
            if typ in ("canal","river","stream","wood","forest","park","nature_reserve","common","towpath"):
                n0 = (namedetails.get("name") or xtra.get("name") or xtra.get("official_name")
                      or (disp.split(",")[0].strip() if disp else ""))
                if n0 and n0.lower() != "unnamed road":
                    out.append(n0)

        for k in ONROUTE_KEYS:
            v = addr.get(k)
            if v and v.strip() and v.lower() != "unnamed road":
                out.append(v.strip())

        nm = namedetails.get("name") or xtra.get("name") or xtra.get("official_name")
        if nm and nm.strip():
            out.append(nm.strip())

        if disp:
            first = disp.split(",")[0].strip()
            if first and first.lower() != "unnamed road":
                out.append(first)
        return out

    names_pos = []  # (name, idx)
    hits = 0
    calls = 0
    N = max(1, len(pts)-1)

    def rank_and_select(pairs):
        best_pos = {}
        for s, i in pairs:
            k = s.strip().lower()
            if not k or k == "unnamed road":
                continue
            if k not in best_pos:
                best_pos[k] = (s.strip(), i)

        scored = []
        for k, (s, i) in best_pos.items():
            l = k
            score = 50.0
            if ("towpath" in l) or ("canal" in l): score -= 30
            if ("wood" in l) or ("woods" in l) or ("forest" in l): score -= 24
            if ("viaduct" in l) or ("bridge" in l): score -= 20
            if any(w in l for w in ["trail","river","park","nature","reserve"]): score -= 15
            if l in BANLIST: score += 25
            if any(w in l for w in ROAD_WORDS) and not any(w in l for w in PRIORITY_WORDS): score += 8
            later = (i / N) if N else 0.0
            score -= 18.0 * later
            scored.append((score, i, s))

        scored.sort()
        return [s for (score, i, s) in scored[:3]]

    for idx, (lat, lon) in enumerate(pts):
        for b in base_order:
            for z in (17, 16):
                if (time.perf_counter() - t0) > BUDGET:
                    break
                calls += 1
                payload = _liq_reverse_cached(b, key, lat, lon, z)
                if payload:
                    cands = extract_names(payload)
                    if cands:
                        for c in cands:
                            names_pos.append((c, idx))
                        hits += 1
                        current = rank_and_select(names_pos)
                        if any(("towpath" in x.lower()) or ("canal" in x.lower()) for x in current) and len(current) >= 3:
                            pois = current
                            rep = {
                                "rid": rid,
                                "points_considered": len(pts),
                                "reverse_calls": calls,
                                "points_hit": hits,
                                "raw_names": [n for n,_ in names_pos][:40],
                                "final_pois": pois,
                                "bases": base_order,
                                "api_key_source": key_src,
                                "early_stop": True
                            }
                            return (pois, rep) if debug else pois
        if (time.perf_counter() - t0) > BUDGET:
            break

    pois = rank_and_select(names_pos)
    rep = {
        "rid": rid,
        "points_considered": len(pts),
        "reverse_calls": calls,
        "points_hit": hits,
        "raw_names": [n for n,_ in names_pos][:40],
        "final_pois": pois,
        "bases": base_order,
        "api_key_source": key_src,
        "early_stop": False
    }
    return (pois, rep) if debug else pois


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

# Warm caches fairly for both routes: reverse lookup (already interleaved), segments, and sentences
try:
    for _r in routes:
        pl = _r.get("polyline") or ""
        _ = cached_onroute_segments(pl, max_pts=220)
        _ = cached_sentence_for_route(_r, max_segments=26)
except Exception:
    pass


# Pre-fetch reverse geocodes for both routes (interleaved) to distribute API calls fairly
try:
    _ = _prefetch_reverse_for_routes(routes, max_pts_each=200)
except Exception:
    pass



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


def _window_dominant_names(names: list[str], window: int = 7, min_dom: float = 0.6) -> list[str]:
    """
    Smooth names by taking the dominant non-blank label in a moving window.
    A name is adopted at position i only if it accounts for >= min_dom of
    the non-blank entries in the window centered at i; otherwise carry forward.
    """
    if not names:
        return []
    n = len(names)
    k = max(1, window//2)
    out = []
    last = (names[0] or "").strip()
    for i in range(n):
        lo = max(0, i-k); hi = min(n, i+k+1)
        win = [(names[j] or "").strip() for j in range(lo, hi)]
        nb = [w for w in win if w]
        if not nb:
            out.append(last)
            continue
        # count
        counts = {}
        for w in nb:
            counts[w.lower()] = counts.get(w.lower(), 0) + 1
        # dominant
        label_lower, cnt = max(counts.items(), key=lambda x: x[1])
        dom = cnt/len(nb)
        if dom >= min_dom:
            # use proper casing from first occurrence in window
            for w in win:
                if w and w.lower() == label_lower:
                    chosen = w
                    break
            last = chosen
        out.append(last)
    return out

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
    MIN_SEG_LEN = 20.0
    MIN_SHARE = 0.006
    strict = []
    # Dynamic easing for short routes
    total = float(tot_len) if tot_len else 0.0
    if total and total < 6000:
        MIN_SEG_LEN = 15.0
        MIN_CONSEC_PTS = 2
        MIN_SHARE = 0.004
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


def _stable_name_sequence(names: list[str], confirm: int = 2) -> list[str]:
    if not names:
        return []
    out = [names[0]]
    cur = names[0]; run = 1
    for i in range(1, len(names)):
        n = names[i] or ""
        if n == cur:
            run += 1; out.append(cur)
        else:
            same = 1
            for j in range(i+1, min(i+confirm, len(names))):
                if names[j] == n: same += 1
                else: break
            if same >= confirm:
                cur = n; run = same; out.append(cur)
            else:
                out.append(cur)
    return out


def onroute_named_segments(polyline: str, *, max_pts: int = 240):
    if not polyline:
        return []
    pts = _sample_points(polyline, max_pts=max_pts)
    if not pts:
        return []
    def haversine(a, b):
        import math
        lat1, lon1 = a; lat2, lon2 = b
        R = 6371000.0
        dlat = math.radians(lat2 - lat1); dlon = math.radians(lon2 - lon1)
        sa = math.sin(dlat/2.0); sb = math.sin(dlon/2.0)
        aa = sa*sa + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*sb*sb
        return 2*R*math.atan2(math.sqrt(aa), math.sqrt(1-aa))
    tot_len = sum(haversine(pts[i-1], pts[i]) for i in range(1, len(pts)))
    raw_names = []
    for (lat, lon) in pts:
        nm = reverse_cache_lookup(lat, lon) or ""
        nm = "" if nm.lower()=="unnamed road" else nm.strip()
        raw_names.append(nm)
    names = _stable_name_sequence(raw_names, confirm=2)
    raw = []
    last = None
    for (latlon, nm) in zip(pts, names):
        nm = nm or ""
        if not raw or (nm != last):
            raw.append({"name": nm or "Unnamed", "coords": [latlon]})
            last = nm
        else:
            raw[-1]["coords"].append(latlon)
    MIN_SEG_LEN = 20.0
    MIN_CONSEC_PTS = 3
    MIN_SHARE = 0.006
    strict = []
    for idx, seg in enumerate(raw):
        nm = (seg["name"] or "").strip()
        if not nm or nm.lower()=="unnamed":
            continue
        coords = seg.get("coords") or []
        if len(coords) < 2:
            continue
        length = sum(haversine(coords[i-1], coords[i]) for i in range(1, len(coords)))
        share = (length/tot_len) if tot_len>0 else 0.0
        if idx in (0, len(raw)-1) or (length >= MIN_SEG_LEN and len(coords) >= MIN_CONSEC_PTS) or share >= MIN_SHARE:
            strict.append({"name": nm, "coords": coords})
    merged = []
    for seg in strict:
        if not merged or merged[-1]["name"].lower() != seg["name"].lower():
            merged.append({"name": seg["name"], "coords": list(seg["coords"])})
        else:
            merged[-1]["coords"].extend(seg["coords"])
    return merged

def describe_turns_sentence(route_dict: dict, *, max_segments: int = 14):
    """Prefer map-matching steps (maneuvers) for Maps-like directions; fallback to on-route segments."""
    steps = locationiq_match_steps(route_dict.get("polyline"))
    segs = steps if steps else onroute_named_segments(route_dict.get("polyline"))
    if not segs:
        return ""

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


@st.cache_data(ttl=7*24*3600, show_spinner=False)
def _cached_directions_sentence(polyline: str | None, url_or_rid: str | None) -> str:
    """Cached wrapper around describe_turns_sentence; keyed by polyline+route id/url."""
    try:
        rd = {"polyline": polyline, "url": url_or_rid, "rid": url_or_rid}
        return describe_turns_sentence(rd)
    except Exception:
        return ""

def cached_sentence_for_route(r: dict) -> str:
    return _cached_directions_sentence(r.get("polyline"), r.get("url") or r.get("rid"))


@st.cache_data(ttl=7*24*3600, show_spinner=False)
def _cached_directions_sentence(polyline: str | None, url_or_rid: str | None, max_segments: int = 26) -> str:
    """Cached wrapper around describe_turns_sentence; keyed by polyline+route id/url."""
    try:
        rd = {"polyline": polyline, "url": url_or_rid, "rid": url_or_rid}
        return describe_turns_sentence(rd, max_segments=max_segments)
    except Exception:
        return ""

def cached_sentence_for_route(r: dict, max_segments: int = 26) -> str:
    return _cached_directions_sentence(r.get("polyline"), r.get("url") or r.get("rid"), max_segments)

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
    sentence = cached_sentence_for_route(r, max_segments=26)

    # Out-and-back neutralisation (keep neutral wording on repeated streets)
    try:
        segs = cached_onroute_segments(r.get("polyline") or "", max_pts=220)
        name_counts = {}
        for seg in segs:
            nm = (seg.get("name") or "").strip()
            if not nm: 
                continue
            k = nm.lower()
            name_counts[k] = name_counts.get(k, 0) + 1
        outback_names = {k for k,v in name_counts.items() if v >= 2}
        # If elevation up/down logic exists, pass outback set to keep those neutral
        try:
            sentence  # keep mypy happy
            _ = _apply_updown  # type: ignore
            # Build minimal grade list if helpers exist
            edge_pts = []
            for seg in segs:
                coords = seg.get("coords") or []
                if coords:
                    edge_pts.extend([coords[0], coords[-1]])
            # Deduplicate while keeping order
            seen = set(); uniq = []
            for p in edge_pts:
                if p not in seen:
                    uniq.append(p); seen.add(p)
            try:
                elevs = _elevations_for_points(uniq)  # type: ignore
            except Exception:
                elevs = None
            if elevs:
                e_map = {uniq[i]: elevs[i] for i in range(len(uniq))}
                grades = [_segment_grade(seg.get("coords") or [], e_map) for seg in segs]  # type: ignore
                sentence = _apply_updown(sentence, segs, grades, thr=0.010, outback_names=outback_names)  # type: ignore
        except Exception:
            # No elevation logic: still neutralise by leaving wording unchanged for repeated names
            pass
    except Exception:
        pass


    # Elevation-aware phrasing (up/down) with out-and-back neutralisation
    try:
        segs = cached_onroute_segments(r.get("polyline") or "", max_pts=220)
        # out-and-back detection: any street name appearing 2+ times
        name_counts = {}
        ordered_names = []
        for seg in segs:
            nm = (seg.get("name") or "").strip()
            if not nm: 
                continue
            ordered_names.append(nm)
            k = nm.lower()
            name_counts[k] = name_counts.get(k, 0) + 1
        outback_names = {k for k,v in name_counts.items() if v >= 2}
        # Collect unique endpoints for elevation calls
        edge_pts = []
        for seg in segs:
            coords = seg.get("coords") or []
            if not coords: 
                continue
            a = coords[0]; b = coords[-1]
            edge_pts.extend([a,b])
        seen = set(); uniq = []
        for p in edge_pts:
            if p not in seen:
                uniq.append(p); seen.add(p)
        try:
            elevs = _elevations_for_points(uniq)  # cached
        except NameError:
            elevs = None
        if elevs:
            e_map = {uniq[i]: elevs[i] for i in range(len(uniq))}
            grades = [_segment_grade(seg.get("coords") or [], e_map) for seg in segs]
            sentence = _apply_updown(sentence, segs, grades, thr=0.010, outback_names=outback_names)
    except Exception:
        pass
    # Elevation-aware phrasing (up/down) — light touch
    try:
        segs = onroute_named_segments(r.get("polyline") or "", max_pts=200)
        # collect unique endpoints for elevation calls
        edge_pts = []
        for seg in segs:
            coords = seg.get("coords") or []
            if not coords: 
                continue
            a = coords[0]; b = coords[-1]
            edge_pts.extend([a,b])
        # dedupe while preserving order
        seen = set(); uniq = []
        for p in edge_pts:
            if p not in seen:
                uniq.append(p); seen.add(p)
        elevs = _elevations_for_points(uniq)
        if elevs:
            e_map = {uniq[i]: elevs[i] for i in range(len(uniq))}
            grades = [_segment_grade(seg.get("coords") or [], e_map) for seg in segs]
            sentence = _apply_updown(sentence, segs, grades, thr=0.010)
    except Exception:
        pass
    if sentence:
        lines.append("  " + sentence)
    return "\n".join(lines)

# Order long/short by distance if available
def sort_with_labels(r1, r2):
    def d(r): return r["dist"] if r["dist"] is not None else -1
    a, b = (r1, r2) if d(r1) >= d(r2) else (r2, r1)
    return [("8k", a), ("5k", b)]


# Auto-preload directions cache for these routes so rendering is fast.
with st.spinner("Preparing directions…"):
    try:
        for _r in routes:
            _ = cached_sentence_for_route(_r)
    except Exception:
        pass

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
if is_road:
    lines.append(SAFETY_NOTE)
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
