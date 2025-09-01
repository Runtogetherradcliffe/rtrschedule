
# pages/social_posts.py
# Build: v2025.09.01-SOCIAL-19 (polished template + emojis + future-date dropdown + copy button)

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
    for (lat, lon) in pts[:10]:  # cap calls
        try:
            resp = requests.get("https://eu1.locationiq.com/v1/reverse",
                                params={"key": key, "lat": lat, "lon": lon, "format": "json"},
                                timeout=10)
            if resp.ok:
                d = resp.json()
                disp = d.get("display_name","")
                # Take first meaningful component
                part = disp.split(",")[0].strip()
                if part and part.lower() not in ("unnamed road",):
                    names.append(part)
        except Exception:
            continue
    # Deduplicate while preserving order
    seen = set(); uniq = []
    for n in names:
        k = n.lower()
        if k not in seen:
            seen.add(k); uniq.append(n)
    return uniq[:3]
from math import radians, sin, cos, asin, sqrt
from datetime import datetime, timezone
from datetime import datetime, timezone, timedelta

def ordinal(n: int) -> str:
    n = int(n)
    if 11 <= (n % 100) <= 13:
        suff = "th"
    else:
        suff = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suff}"


def format_day_month_uk(dts):
    ts = pd.Timestamp(dts)
    return f"{ordinal(ts.day)} {ts.strftime('%B')}"
def format_full_uk_date(d: pd.Timestamp) -> str:
    # Expect naive datetime64[ns]; convert to Timestamp if needed
    ts = pd.Timestamp(d)
    return f"{ts.strftime('%A')} {ordinal(ts.day)} {ts.strftime('%B')}"



# Try to import shared settings (optional)
try:
    from app_config import get_cfg
except Exception:
    def get_cfg(k, default=""):
        return st.session_state.get(k, default)

st.set_page_config(page_title="Weekly Social Post Composer", page_icon=":mega:", layout="wide")
st.title("Weekly Social Post Composer")
st.caption("Build: v2025.09.01-SOCIAL-19 — polished template, emoji rules, future-date picker, clipboard.")


# ----------------------------- Helpers ----------------------------------

from datetime import timedelta
def to_thursday_date(ts):
    try:
        d = pd.to_datetime(ts, errors="coerce", dayfirst=True)
    except Exception:
        d = pd.to_datetime(ts, errors="coerce")
    if pd.isna(d):
        return None
    # Ensure naive datetime (date-only OK)
    try:
        d = d.tz_localize(None)
    except Exception:
        pass
    # 0=Mon ... 3=Thu
    wd = int(d.weekday())
    offset = (3 - wd) % 7
    return (d + pd.to_timedelta(offset, unit="D"))
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

STRAVA_ROUTE_ID_RE = re.compile(r"(?:^|/)(?:routes|routes/view)/(\\d+)(?:[/?#].*)?$", re.I)

def is_strava_route_url(u: str) -> bool:
    if not u:
        return False
    lu = u.lower()
    return ("strava.com" in lu) and ("/routes/" in lu)

def extract_route_id_from_url(u: str) -> str:
    m = STRAVA_ROUTE_ID_RE.search(u or "")
    return m.group(1) if m else ""

def extract_digits(s: str) -> str:
    return "".join(re.findall(r"\\d+", s or ""))

def expand_sci_id(s: str) -> str:
    if not s:
        return ""
    s = s.strip()
    m = re.match(r"^(\\d+)(?:\\.(\\d+))?[eE]\\+?(\\d+)$", s)
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

# Hilliness descriptors with variation
def hilliness_blurb(dist_km, elev_m):
    phrases = {
        "flat": ["flat and friendly 🏁", "fast & flat 🏁", "pan-flat cruise 💨"],
        "rolling": ["gently rolling 🌱", "undulating and friendly 🌿", "rolling countryside vibes 🌳"],
        "hilly": ["a hilly tester! ⛰️", "spicy climbs ahead 🌶️", "some punchy hills 🚵"],
    }
    if not dist_km or not elev_m:
        return random.choice(["a great midweek spin", "perfect for all paces", "midweek miles made easy"])
    try:
        m_per_km = float(elev_m)/max(float(dist_km), 0.1)
    except Exception:
        return random.choice(["a great midweek spin", "perfect for all paces", "midweek miles made easy"])
    key = "flat" if m_per_km < 10 else ("rolling" if m_per_km < 20 else "hilly")
    return random.choice(phrases[key])
    try:
        m_per_km = float(elev_m)/max(float(dist_km), 0.1)
    except Exception:
        return random.choice(["a great midweek spin", "perfect for all paces", "midweek miles made easy"])
    if m_per_km < 5:
        return random.choice(phrases["flat"])
    if m_per_km < 15:
        return random.choice(phrases["rolling"])
    return random.choice(phrases["hilly"])

# Clipboard helper (JS injection)
def copy_button(label, text, key):
    escaped = text.replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$").replace("</", "<\\/")
    btn_key = f"copy_btn_{key}"
    html = f'''
    <button id="{btn_key}" style="padding:6px 10px; border:1px solid #ddd; border-radius:6px; cursor:pointer;">📋 {label}</button>
    <script>
    const btn = document.getElementById("{btn_key}");
    if (btn) {{
        btn.addEventListener("click", async () => {{
            try {{
                await navigator.clipboard.writeText(`{escaped}`);
                btn.innerText = "✅ Copied!";
                setTimeout(() => btn.innerText = "📋 {label}", 1500);
            }} catch (e) {{
                btn.innerText = "❌ Copy failed";
                setTimeout(() => btn.innerText = "📋 {label}", 1500);
            }}
        }});
    }}
    </script>
    '''
    st.markdown(html, unsafe_allow_html=True)


# --------------------------- UX Settings ---------------------------------
st.sidebar.header("Settings (session)")
gs_default = st.sidebar.text_input("Default Google Sheet URL", value=get_cfg("GS_URL_DEFAULT"))
if st.sidebar.button("Save as session default"):
    st.session_state["GS_URL_DEFAULT"] = gs_default
    st.sidebar.success("Saved for this session.")

# ---------------------------- Input controls -----------------------------
st.subheader("Pick a future date")
gs_url = st.text_input("Google Sheet URL", value=get_cfg("GS_URL_DEFAULT"))

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

date_col = "Date (Thu)"

meet_loc_col = None
for c in sched.columns:
    if "meet" in c.lower() and "loc" in c.lower():
        meet_loc_col = c
        break
notes_col = "Notes"

r_names = ["Route 1 - Name", "Route 2 - Name"]
r_urls = ["Route 1 - Route Link (Source URL)", "Route 2 - Route Link (Source URL)"]
r_srcid = ["Route 1 - Source ID", "Route 2 - Source ID"]
r_terrain = ["Route 1 - Terrain (Road/Trail/Mixed)", "Route 2 - Terrain (Road/Trail/Mixed)"]
r_area = ["Route 1 - Area", "Route 2 - Area"]
r_dist = ["Route 1 - Distance (km)", "Route 2 - Distance (km)"]
r_elev = [None, None]
r_pois = [None, None]

if not date_col or not all(r_names) or not all(r_urls):
    st.error("Missing required columns in Schedule (Date, Route Names, Route URLs).")
    st.stop()

# --- Future-only dropdown using pure date (no tz) ---
d = pd.to_datetime(sched[date_col], errors="coerce", format="%Y-%m-%d %H:%M:%S")
sched["_dateonly"] = d.dt.date
today = pd.Timestamp.today().date()
future_rows = sched[sched["_dateonly"] >= today]
# Only Thursdays (Mon=0..Thu=3)
future_rows = future_rows[pd.to_datetime(future_rows["_dateonly"]).dt.weekday == 3]
future_rows = future_rows.sort_values("_dateonly")
opt_idx = future_rows.index.tolist()
def _fmt(idx):
    return format_day_month_uk(pd.to_datetime(future_rows.loc[idx, "_dateonly"]))
idx_choice = st.selectbox("Date", options=opt_idx, format_func=_fmt, index=0 if opt_idx else None)
if idx_choice is None:
    st.stop()
row = future_rows.loc[idx_choice]



def try_float(s):
    """Extract first numeric value from strings like '8km', '1,200 m', '75m', or plain numbers."""
    if s is None:
        return None
    try:
        if isinstance(s, (int, float)):
            return float(s)
        t = str(s)
        t = t.replace(',', '')
        m = re.search(r"[-+]?\d*\.?\d+", t)
        return float(m.group(0)) if m else None
    except Exception:
        return None

def make_https(u):
    u = (u or "").strip()
    if not u:
        return u
    if " " in u or u.lower().startswith("strava route"):
        return ""  # placeholder/invalid
    if not urllib.parse.urlparse(u).scheme:
        return "https://" + u
    return u

def extract_route_id(url: str, source_id: str) -> str:
    m = STRAVA_ROUTE_ID_RE.search(url or "")
    if m:
        return m.group(1)
    expanded = expand_sci_id(source_id)
    return expanded if expanded else extract_digits(source_id)

def build_route_dict(side_idx: int):
    nm = clean(row.get(r_names[side_idx], ""))
    url_raw = clean(row.get(r_urls[side_idx], ""))
    sid_raw = clean(row.get(r_srcid[side_idx], "")) if r_srcid[side_idx] else ""
    terr = clean(row.get(r_terrain[side_idx], "")) if r_terrain[side_idx] else ""
    area = clean(row.get(r_area[side_idx], "")) if r_area[side_idx] else ""
    url = make_https(url_raw)
    rid = extract_route_id(url, sid_raw) if (url or sid_raw) else ""
    dist = try_float(row.get(r_dist[side_idx], "")) if r_dist[side_idx] else None
    elev = try_float(row.get(r_elev[side_idx], "")) if r_elev[side_idx] else None
    pois = clean(row.get(r_pois[side_idx], "")) if r_pois[side_idx] else ""
    return {"name": nm, "url": url, "rid": rid, "terrain": terr, "area": area,
            "dist": dist, "elev": elev, "pois": pois}

routes = [build_route_dict(0), build_route_dict(1)]
# Try to enrich metrics & highlights from Strava + LocationIQ
for r in routes:
    if (r.get("dist") is None or r.get("elev") is None or not r.get("pois")) and (r.get("url") or r.get("rid")):
        met = fetch_strava_route_metrics(r.get("url") or r.get("rid"))
        if met:
            if r.get("dist") is None and met.get("dist_km"):
                r["dist"] = met["dist_km"]
            if r.get("elev") is None and met.get("elev_m") is not None:
                r["elev"] = met["elev_m"]
            if not r.get("pois"):
                hi = fetch_locationiq_highlights(met.get("polyline"))
                if hi:
                    r["pois"] = ", ".join(hi)

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
    line2 = f"  {dist_txt} – {desc}"
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