
# helpers/strava_gpx.py (v2 - fixed columns & sheet)
import io
import re
import time
import urllib.parse
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st

STRAVA_SHEET_ID = "1ncT1NCbSnFsAokyFBkMWBVsk7yrJTiUfG0iBRxyUCTw"
SHEET_TAB = "Annual_Schedule_MASTER"

# ---------- OAuth capture ----------
def capture_strava_token_from_query():
    if st.session_state.get("strava_token"):
        return
    code = st.query_params.get("code")
    if not code:
        return
    client_id = st.secrets.get("STRAVA_CLIENT_ID")
    client_secret = st.secrets.get("STRAVA_CLIENT_SECRET")
    if not client_id or not client_secret:
        st.warning("Strava 'code' detected but API secrets are missing in Streamlit Secrets.")
        return
    token_url = "https://www.strava.com/oauth/token"
    payload = {
        "client_id": client_id,
        "client_secret": client_secret,
        "code": code,
        "grant_type": "authorization_code",
    }
    try:
        resp = requests.post(token_url, data=payload, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        st.session_state["strava_token"] = data.get("access_token")
        st.session_state["strava_athlete"] = data.get("athlete", {})
        qp = dict(st.query_params)
        if "code" in qp:
            del qp["code"]
            st.query_params.clear()
            for k, v in qp.items():
                st.query_params[k] = v
        st.success("Strava connected")
    except Exception as e:
        st.error(f"Strava token exchange failed: {e}")

# ---------- Sheet loading ----------
def load_google_sheet_csv(sheet_id: str, sheet_name: str) -> pd.DataFrame:
    u = (
        f"https://docs.google.com/spreadsheets/d/{sheet_id}"
        "/gviz/tq?tqx=out:csv&sheet=" + urllib.parse.quote(sheet_name, safe="")
    )
    df = pd.read_csv(u, dtype=str, keep_default_na=False)
    return df

def load_master_schedule() -> pd.DataFrame:
    try:
        df = load_google_sheet_csv(STRAVA_SHEET_ID, SHEET_TAB)
        return df
    except Exception as e:
        st.error(f"Could not load Google Sheet: {e}")
        return pd.DataFrame()

# ---------- Route ID parsing ----------
STRAVA_ROUTE_ID_RE = re.compile(r"(?:^|/)(?:routes|routes/view)/(\d+)(?:[/?#].*)?$", re.I)

def extract_route_id_from_url(u: str) -> str:
    m = STRAVA_ROUTE_ID_RE.search(u or "")
    return m.group(1) if m else ""

def digits_only(s: str) -> str:
    return "".join(re.findall(r"\d+", s or ""))

def expand_scientific_to_digits(s: str) -> str:
    if not s or not re.search(r"e\+?\-?\d+", str(s), re.I):
        return ""
    m = re.match(r"^\s*([0-9]+)(?:\.([0-9]+))?[eE]\+?(-?\d+)\s*$", str(s))
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

def choose_route_id_from_cells(url_val: str, source_id_val: str) -> str:
    rid = extract_route_id_from_url(url_val or "")
    if rid:
        return rid
    sid = str(source_id_val or "").strip()
    if not sid:
        return ""
    d = digits_only(sid)
    if len(d) >= 6:
        return d
    exp = expand_scientific_to_digits(sid)
    if exp and exp.isdigit():
        return exp
    return ""

# ---------- Bucketing (Route1=8k, Route2=5k; Mixed -> Trail) ----------
@dataclass
class BucketedRoute:
    route_id: str
    group: str   # '5k' or '8k'
    surface: str # 'Road' or 'Trail'

ROUTE1_COLS = {
    "terrain": "Route 1 - Terrain (Road/Trail/Mixed)",
    "link_type": "Route 1 - Route Link Type",
    "url": "Route 1 - Route Link (Source URL)",
    "source_id": "Route 1 - Source ID",
}
ROUTE2_COLS = {
    "terrain": "Route 2 - Terrain (Road/Trail/Mixed)",
    "link_type": "Route 2 - Route Link Type",
    "url": "Route 2 - Route Link (Source URL)",
    "source_id": "Route 2 - Source ID",
}

def norm_surface(v: str) -> Optional[str]:
    v = (v or "").strip().lower()
    if "road" in v:
        return "Road"
    if "trail" in v or "mixed" in v:
        return "Trail"  # Mixed -> Trail (as agreed)
    return None

def add_row_to_bucket(row: dict, cols: dict, fixed_group: str, buckets: Dict[str, List[BucketedRoute]], errors: List[str]):
    link_type = str(row.get(cols["link_type"], "")).strip().lower()
    if link_type != "strava route":
        return
    rid = choose_route_id_from_cells(row.get(cols["url"], ""), row.get(cols["source_id"], ""))
    surf = norm_surface(row.get(cols["terrain"], ""))
    if not rid or not surf:
        return
    key = f"{surf} {fixed_group}"
    buckets.setdefault(key, []).append(BucketedRoute(route_id=rid, group=fixed_group, surface=surf))

def build_buckets_from_master(df: pd.DataFrame) -> Tuple[Dict[str, List[BucketedRoute]], List[str]]:
    buckets: Dict[str, List[BucketedRoute]] = {
        "Road 5k": [], "Trail 5k": [], "Road 8k": [], "Trail 8k": []
    }
    errors: List[str] = []

    # Validate required columns exist
    required_cols = set(ROUTE1_COLS.values()) | set(ROUTE2_COLS.values())
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        errors.append("Missing expected columns: " + ", ".join(missing))
        return buckets, errors

    for _, row in df.iterrows():
        # Route 1 -> 8k
        add_row_to_bucket(row, ROUTE1_COLS, "8k", buckets, errors)
        # Route 2 -> 5k
        add_row_to_bucket(row, ROUTE2_COLS, "5k", buckets, errors)

    return buckets, errors

# ---------- Strava download ----------
def strava_export_gpx(route_id: str, token: str) -> bytes:
    url = f"https://www.strava.com/api/v3/routes/{route_id}/export_gpx"
    r = requests.get(url, headers={"Authorization": f"Bearer {token}" }, timeout=30)
    if r.status_code == 429:
        time.sleep(2)
        r = requests.get(url, headers={"Authorization": f"Bearer {token}" }, timeout=30)
    r.raise_for_status()
    return r.content

# ---------- Zipping with dedupe ----------
def build_bucket_zips(buckets: Dict[str, List[BucketedRoute]], token: str):
    import zipfile
    import io as _io

    # Unique route IDs across all buckets
    unique_ids = []
    for blist in buckets.values():
        for br in blist:
            if br.route_id not in unique_ids:
                unique_ids.append(br.route_id)

    gpx_cache = {}
    errors_by_route: Dict[str, str] = {}

    progress = st.progress(0.0)
    status = st.empty()
    total = len(unique_ids)
    for idx, rid in enumerate(unique_ids, start=1):
        status.write(f"Fetching GPX {idx}/{total} (route {rid}) …")
        try:
            gpx_cache[rid] = strava_export_gpx(rid, token)
        except Exception as e:
            gpx_cache[rid] = None
            errors_by_route[rid] = str(e)
        progress.progress(idx / max(total, 1))

    zips = {}
    for bucket_name, blist in buckets.items():
        bio = _io.BytesIO()
        with zipfile.ZipFile(bio, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for br in blist:
                g = gpx_cache.get(br.route_id)
                if not g:
                    errors_by_route[f"{bucket_name}:{br.route_id}"] = errors_by_route.get(br.route_id, "Unknown error")
                    continue
                zf.writestr(f"strava_route_{br.route_id}.gpx", g)
        zips[bucket_name] = bio.getvalue()

    status.write("Done.")
    return zips, errors_by_route
