
# pages/social_posts.py
# Build: v2025.09.01-SOCIAL-22.1 (polished template + emojis + future-date dropdown + copy button)

import io
import re
import time
import random
import urllib.parse
import pandas as pd
import requests
import streamlit as st

# === Strava & LocationIQ enrichment helpers ===
import json, math

def _get_secret(name, default=None):
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default

def get_strava_token():
    # Prefer session token set by OAuth pages
    for k in ("strava_access_token", "strava_token", "access_token"):
        tok = st.session_state.get(k)
        if tok:
            return tok
    # Optional: refresh via secrets if refresh token provided
    cid = _get_secret("STRAVA_CLIENT_ID")
    csec = _get_secret("STRAVA_CLIENT_SECRET")
    rtok = st.session_state.get("strava_refresh_token") or _get_secret("STRAVA_REFRESH_TOKEN")
    if cid and csec and rtok:
        try:
            resp = requests.post("https://www.strava.com/oauth/token", data={
                "client_id": cid, "client_secret": csec,
                "grant_type": "refresh_token", "refresh_token": rtok
            }, timeout=15)
            if resp.ok:
                data = resp.json()
                st.session_state["strava_access_token"] = data.get("access_token")
                st.session_state["strava_refresh_token"] = data.get("refresh_token", rtok)
                return data.get("access_token")
        except Exception:
            pass
    return None

def _extract_route_id(url):
    # Accepts full strava route URL or digits
    if not url:
        return None
    s = str(url)
    m = re.search(r"/routes/(\d+)", s)
    if m: return m.group(1)
    # fallback: digits only
    m = re.search(r"(\d{6,})", s)
    return m.group(1) if m else None

def fetch_strava_route_metrics(url_or_id):
    rid = _extract_route_id(url_or_id)
    if not rid:
        return None
    tok = get_strava_token()
    if not tok:
        return None
    try:
        r = requests.get(f"https://www.strava.com/api/v3/routes/{rid}",
                         headers={"Authorization": f"Bearer {tok}"}, timeout=15)
        if not r.ok:
            return None
        j = r.json()
        dist_km = float(j.get("distance", 0.0))/1000.0 if j.get("distance") is not None else None
        elev_m = j.get("elevation_gain")
        try:
            elev_m = float(elev_m) if elev_m is not None else None
        except Exception:
            elev_m = None
        poly = None
        if j.get("map") and (j["map"].get("polyline") or j["map"].get("summary_polyline")):
            poly = j["map"].get("polyline") or j["map"].get("summary_polyline")
        return {"dist_km": dist_km, "elev_m": elev_m, "polyline": poly}
    except Exception:
        return None

# Simple polyline decoder (Google polyline algorithm)
def _decode_polyline(polyline_str):
    if not polyline_str:
        return []
    index, lat, lng = 0, 0, 0
    coords = []
    while index < len(polyline_str):
        result, shift = 0, 0
        while True:
            b = ord(polyline_str[index]) - 63
            index += 1
            result |= (b & 0x1f) << shift
            shift += 5
            if b < 0x20:
                break
        dlat = ~(result >> 1) if (result & 1) else (result >> 1)
        lat += dlat
        result, shift = 0, 0
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

def sample_points(poly, every= max(1, 30)):
    pts = _decode_polyline(poly)
    if not pts: return []
    if len(pts) <= every: return pts
    return [pts[i] for i in range(0, len(pts), every)]

def fetch_locationiq_highlights(polyline):
    key = _get_secret("LOCATIONIQ_API_KEY") or _get_secret("LOCATIONIQ_TOKEN")
    if not key or not polyline:
        return []
    pts = sample_points(polyline, every=40)
    names = []
    for (lat, lon) in pts[:10]:
        try:
            resp = requests.get("https://eu1.locationiq.com/v1/reverse",
                                params={"key": key, "lat": lat, "lon": lon, "format": "json"},
                                timeout=10)
            if resp.ok:
                d = resp.json()
                disp = d.get("display_name","")
                part = disp.split(",")[0].strip()
                if part and part.lower() not in ("unnamed road",):
                    names.append(part)
        except Exception:
            continue
    seen=set(); uniq=[]
    for n in names:
        k=n.lower()
        if k not in seen:
            seen.add(k); uniq.append(n)
    return uniq[:3]
# === Enrichment cache ===
if "enrich_cache" not in st.session_state:
    st.session_state["enrich_cache"] = {}

def _cache_get(rid):
    return st.session_state["enrich_cache"].get(str(rid))

def _cache_put(rid, data):
    st.session_state["enrich_cache"][str(rid)] = data

# --- GPX fallback utilities ---
import xml.etree.ElementTree as ET
import math

def _haversine(lat1, lon1, lat2, lon2):
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dl/2)**2
    return 2*R*math.asin(math.sqrt(a))

def _parse_gpx(gpx_bytes):
    try:
        root = ET.fromstring(gpx_bytes)
    except Exception:
        return []
    ns = {"gpx": "http://www.topografix.com/GPX/1/1"}
    pts = []
    for trkpt in root.findall(".//gpx:trkpt", ns):
        lat = trkpt.get("lat"); lon = trkpt.get("lon")
        if lat is None or lon is None:
            continue
        ele_el = trkpt.find("gpx:ele", ns)
        ele = float(ele_el.text) if ele_el is not None else None
        pts.append((float(lat), float(lon), ele))
    return pts

def _metrics_from_points(pts):
    if len(pts) < 2:
        return None, None
    dist_m = 0.0
    gain = 0.0
    prev = pts[0]
    for cur in pts[1:]:
        dist_m += _haversine(prev[0], prev[1], cur[0], cur[1])
        if prev[2] is not None and cur[2] is not None:
            delta = cur[2] - prev[2]
            if delta > 0:
                gain += delta
        prev = cur
    return dist_m/1000.0, gain

def fetch_strava_route_gpx_points(route_id, token):
    try:
        r = requests.get(f"https://www.strava.com/api/v3/routes/{route_id}/export_gpx",
                         headers={"Authorization": f"Bearer {token}"},
                         timeout=20)
        if not r.ok:
            return []
        pts = _parse_gpx(r.content)
        return pts
    except Exception:
        return []

def enrich_route(r):
    rid = _extract_route_id(r.get("url") or r.get("rid"))
    if not rid:
        return r, {"source": "none", "note": "no route id/url"}

    cached = _cache_get(rid)
    if cached:
        if cached.get("dist_km") is not None:
            r["dist"] = cached["dist_km"]
        if cached.get("elev_m") is not None and r.get("elev") is None:
            r["elev"] = cached["elev_m"]
        if not r.get("pois") and cached.get("pois"):
            r["pois"] = ", ".join(cached["pois"])
        return r, dict(cached, source="cache")

    tok = get_strava_token()
    met = fetch_strava_route_metrics(r.get("url") or r.get("rid")) if tok else None
    polyline = met.get("polyline") if met else None
    dist_km = met.get("dist_km") if met else None
    elev_m  = met.get("elev_m")  if met else None
    source = "strava_meta" if met else "none"

    sample = []
    if tok and (dist_km is None or elev_m is None or not polyline):
        pts = fetch_strava_route_gpx_points(rid, tok)
        if pts:
            d_km, g_m = _metrics_from_points(pts)
            if dist_km is None and d_km:
                dist_km = d_km
                source = "gpx_fallback"
            if elev_m is None and g_m is not None:
                elev_m = g_m
                source = "gpx_fallback"
            if not polyline:
                step = max(1, len(pts)//40)
                sample = [(lat, lon) for (lat,lon,ele) in pts[::step]]

    pois = []
    key = _get_secret("LOCATIONIQ_API_KEY") or _get_secret("LOCATIONIQ_TOKEN")
    if key:
        if polyline:
            pois = fetch_locationiq_highlights(polyline)
        elif sample:
            names = []
            for (lat, lon) in sample[:10]:
                try:
                    resp = requests.get(
                        "https://eu1.locationiq.com/v1/reverse",
                        params={"key": key, "lat": lat, "lon": lon, "format": "json"},
                        timeout=10,
                    )
                    if resp.ok:
                        d = resp.json()
                        first = (d.get("display_name","").split(",")[0]).strip()
                        if first and first.lower() not in ("unnamed road",):
                            names.append(first)
                except Exception:
                    continue
            seen=set(); uniq=[]
            for n in names:
                k=n.lower()
                if k not in seen:
                    seen.add(k); uniq.append(n)
            pois = uniq[:3]

    if dist_km is not None:
        r["dist"] = dist_km
    if r.get("elev") is None and elev_m is not None:
        r["elev"] = elev_m
    if not r.get("pois") and pois:
        r["pois"] = ", ".join(pois)

    _cache_put(rid, {"dist_km": dist_km, "elev_m": elev_m, "pois": pois, "source": source})
    return r, {"dist_km": dist_km, "elev_m": elev_m, "pois": pois, "source": source}

routes = [build_route_dict(0), build_route_dict(1)]
enrich_reports = []
for r in routes:
    if (r.get("url") or r.get("rid")):
        r, rep = enrich_route(r)
        enrich_reports.append(rep)
    else:
        enrich_reports.append({"source":"none","note":"no url"})
with st.expander("Debug: enrichment", expanded=False):
    try:
        st.json(enrich_reports)
    except Exception:
        st.write(enrich_reports)

for r in routes:
    if (r.get("url") or r.get("rid")):
        r, rep = enrich_route(r)
        enrich_reports.append(rep)
    else:
        enrich_reports.append({"source":"none","note":"no url"})
with st.expander("Debug: enrichment", expanded=False):
    try:
        st.json(enrich_reports)
    except Exception:
        st.write(enrich_reports)

# Label longer = 8k, shorter = 5k (if distances available)
def sort_with_labels(r1, r2):
    def d(r): return r["dist"] if r["dist"] is not None else -1
    a, b = (r1, r2) if d(r1) >= d(r2) else (r2, r1)
    return [("8k", a), ("5k", b)]
labeled = sort_with_labels(routes[0], routes[1])

# Event detection
meet_loc = clean(row.get(meet_loc_col, "")) if meet_loc_col else ""
# MEETING_PARSE_INJECTED
if not meet_loc:
    _notes = str(row.get("Notes", ""))
    m = re.search(r"Meeting:\s*([^|\n]+)", _notes, re.IGNORECASE)
    if m:
        meet_loc = m.group(1).strip()
if not meet_loc:
    meet_loc = get_cfg("MEET_LOC_DEFAULT", "Radcliffe Market")
notes = clean(row.get(notes_col, "")) if notes_col else ""

def has_kw(s, *kws):
    s = (s or "").lower()
    return any(kw in s for kw in kws)

is_on_tour = has_kw(notes, "on tour", "ontour") or (meet_loc and meet_loc.lower() != "radcliffe market")
is_pride = has_kw(notes, "pride", "rainbow", "lgbt", "🏳️‍🌈")
has_social = has_kw(notes, "social", "pub", "after")

# Build friendly copy
date_str = format_day_month_uk(pd.to_datetime(row["_dateonly"]))
header = "🌈 Pride Run!" if is_pride else ("🚌 On Tour!" if is_on_tour else "🏃 This Thursday")
meeting_line = f"📍 Meeting at: {(meet_loc or get_cfg('MEET_LOC_DEFAULT', 'Radcliffe Market')).title()}"
time_line = "🕖 We set off at 7:00pm"

def route_blurb(label, r):
    dist_txt = f"{r['dist']:.1f} km" if isinstance(r["dist"], (int, float)) else (f"{r['dist']} km" if r["dist"] is not None else "? km")
    desc = hilliness_blurb(r["dist"], r["elev"])
    url = r["url"] or (f"https://www.strava.com/routes/{r['rid']}" if r["rid"] else "")
    name = r["name"] or "Route"
    line1 = f"• {label} – {name}" + (f": {url}" if url else "")
    line2 = f"  {dist_txt}" + (f" with {r['elev']:.0f}m of elevation" if isinstance(r.get("elev"), (int, float)) else "") + f" – {desc}"
    highlights = ""
    if r.get("pois"):
        chunks = [c.strip() for c in str(r["pois"]).split("|") if c.strip()]
        parts = []
        for ch in chunks:
            parts.extend([p.strip() for p in re.split(r"[;,]", ch) if p.strip()])
        if parts:
            highlights = "🏞️ Highlights: " + ", ".join(parts[:3])
    lines = [line1, line2]
    if highlights:
        lines.append("  " + highlights)
    return "\n".join(lines)

lines = []
lines.append(f"{header} — {date_str}")
lines.append("")
lines.append(meeting_line)
lines.append(time_line)
lines.append("")
lines.append("🛣️ This week we’ve got two route options to choose from:")
for label, r in labeled:
    lines.append(route_blurb(label, r))

if has_social:
    lines.append("")
    lines.append("🍻 Afterwards, we're heading for a post-run social — come along!")

lines.append("")
lines.append("📲 Book now:")
lines.append(get_cfg("BOOK_URL", "https://groups.runtogether.co.uk/RunTogetherRadcliffe/Runs"))
lines.append("❌ Can’t make it? Cancel at least 1 hour before:")
lines.append(get_cfg("CANCEL_URL", "https://groups.runtogether.co.uk/My/BookedRuns"))
lines.append("")
lines.append("👟 Grab your shoes, bring your smiles – see you Thursday!")
lines.append("")
lines.append("*RunTogether Radcliffe – This Thursday!*")

post_text = "\n".join(lines)

st.subheader("Composed message")
st.text_area("Copy/paste to socials", value=post_text, height=420, key="long_post_area")
copy_button("Copy message", st.session_state.get("long_post_area", post_text), key="long")

# Short post variant
short = "{date}: {n1} ({d1}km) & {n2} ({d2}km). Meet {meet} {time}. {u1} {u2}".format(
    date=date_str,
    n1=labeled[0][1]['name'], d1=(labeled[0][1]['dist'] or ''),
    n2=labeled[1][1]['name'], d2=(labeled[1][1]['dist'] or ''),
    meet=(meet_loc or 'Radcliffe Market'),
    time=get_cfg('MEET_TIME_DEFAULT', '19:00'),
    u1=( (labeled[0][1]['url']) or (f"https://www.strava.com/routes/{labeled[0][1]['rid']}") ),
    u2=( (labeled[1][1]['url']) or (f"https://www.strava.com/routes/{labeled[1][1]['rid']}") )
)
st.text_area("Short post", value=short, height=110, key="short_post_area")
copy_button("Copy short", st.session_state.get("short_post_area", short), key="short")