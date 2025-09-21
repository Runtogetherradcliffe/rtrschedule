import os
import json
import math
from typing import List, Dict, Any, Tuple
import requests
import streamlit as st

RADAR_BASE = "https://api.radar.io/v1"
RADAR_API_KEY = os.getenv("RADAR_API_KEY", "") or st.secrets.get("RADAR_API_KEY", "")

def _decode_polyline(polyline_str: str) -> List[Tuple[float, float]]:
    if not polyline_str:
        return []
    coords = []
    index = lat = lon = 0
    length = len(polyline_str)
    while index < length:
        result = 1
        shift = 0
        while True:
            b = ord(polyline_str[index]) - 63 - 1
            index += 1
            result += b << shift
            shift += 5
            if b < 0x1f:
                break
        dlat = ~(result >> 1) if result & 1 else (result >> 1)
        lat += dlat

        result = 1
        shift = 0
        while True:
            b = ord(polyline_str[index]) - 63 - 1
            index += 1
            result += b << shift
            shift += 5
            if b < 0x1f:
                break
        dlng = ~(result >> 1) if result & 1 else (result >> 1)
        lon += dlng
        coords.append((lat * 1e-5, lon * 1e-5))
    return coords

def _resample(points: List[Tuple[float, float]], max_pts: int = 300) -> List[Tuple[float, float]]:
    if not points:
        return points
    n = len(points)
    if n <= max_pts:
        return points
    step = max(1, math.floor(n / max_pts))
    out = points[::step]
    if out[-1] != points[-1]:
        out.append(points[-1])
    return out

def _canonical_name(nm: str) -> str:
    if not nm: return ""
    import re
    s = re.sub(r"^[A-Z]{1,2}\d+\s+", "", nm.strip())
    s = re.sub(r"\brd\b|\brd\.\b", "Road", s, flags=re.IGNORECASE)
    s = re.sub(r"\bln\b|\bln\.\b", "Lane", s, flags=re.IGNORECASE)
    s = re.sub(r"\bave\b|\bave\.\b", "Avenue", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _unique_first(segments: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
    seen = set(); out = []
    for seg in segments:
        nm = _canonical_name(seg.get("name") or "")
        k = nm.lower()
        if k and k not in seen:
            seen.add(k)
            out.append({"name": nm, "coords": seg.get("coords") or seg.get("geometry") or []})
    return out

def _coords_to_geojson(points: List[Tuple[float, float]]) -> Dict[str,Any]:
    return {"type":"LineString", "coordinates":[[p[1], p[0]] for p in points]}

@st.cache_data(ttl=180*24*3600, show_spinner=False)
def radar_match_steps(polyline_str: str, mode: str = "foot") -> List[Dict[str,Any]]:
    if not RADAR_API_KEY:
        raise RuntimeError("RADAR_API_KEY is not configured in env or Streamlit secrets.")
    pts = _resample(_decode_polyline(polyline_str), max_pts=300)
    if not pts:
        return []
    geojson = _coords_to_geojson(pts)
    headers = {"Authorization": RADAR_API_KEY, "Content-Type":"application/json"}

    url = f"{RADAR_BASE}/route/match"
    payload = {"mode": mode, "geometry": geojson}
    r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=20)
    if r.status_code == 404:
        url = f"{RADAR_BASE}/route/directions"
        payload = {"mode": mode, "geometry": geojson, "steps": True}
        r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=20)
    r.raise_for_status()
    data = r.json()

    segments = []
    routes = data.get("routes") or []
    if routes:
        for leg in routes[0].get("legs", []):
            for s in leg.get("steps", []) or []:
                name = s.get("name") or s.get("road") or ""
                geom = s.get("geometry") or {}
                coords = geom.get("coordinates") or []
                if name:
                    segments.append({"name": _canonical_name(name), "coords": coords})
    return _unique_first(segments)
