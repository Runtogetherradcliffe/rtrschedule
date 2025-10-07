
# helpers/strava_gpx.py
# Utilities for: OAuth capture, schedule loading, route-id extraction, Strava GPX download, zipping.
# Designed to be self-contained for easy drop-in.

import io
import re
import time
import urllib.parse
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st

# ---------------- OAuth capture (re-uses the pattern already in your project) ----------------
def capture_strava_token_from_query():
    """
    If this page is a redirect target and has ?code=..., exchange it for a token and
    stash token & athlete in session_state.
    """
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
        # Clean the query params so the code isn't visible after exchange
        qp = dict(st.query_params)
        if "code" in qp:
            del qp["code"]
            st.query_params.clear()
            for k, v in qp.items():
                st.query_params[k] = v
        st.success("Strava connected")
    except Exception as e:
        st.error(f"Strava token exchange failed: {e}")

# ---------------- Schedule loading ----------------

def norm_header(h: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(h).strip().lower())

def extract_sheet_id(url: str) -> Optional[str]:
    m = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", url or "")
    return m.group(1) if m else None

def load_google_sheet_csv(sheet_id: str, sheet_name: str) -> pd.DataFrame:
    u = (
        f"https://docs.google.com/spreadsheets/d/{sheet_id}"
        "/gviz/tq?tqx=out:csv&sheet=" + urllib.parse.quote(sheet_name, safe="")
    )
    df = pd.read_csv(u, dtype=str, keep_default_na=False)
    return df

def load_from_google_csv(url: str) -> Dict[str, pd.DataFrame]:
    """Try to load at least the 'Schedule' tab from a public Google Sheet link."""
    sid = extract_sheet_id(url)
    dfs: Dict[str, pd.DataFrame] = {}
    if not sid:
        return dfs
    try:
        sched = load_google_sheet_csv(sid, "Schedule")
        if not sched.empty:
            dfs["Schedule"] = sched
        # Try to load optional helper tabs if present
        for tab in ["Route Master", "RouteMaster", "Routemaster", "Config"]:
            try:
                df = load_google_sheet_csv(sid, tab)
                if not df.empty:
                    dfs[tab] = df
            except Exception:
                pass
    except Exception as e:
        st.error(f"Could not load Google Sheet CSVs: {e}")
    return dfs

def load_from_uploaded_excel(uploaded) -> Dict[str, pd.DataFrame]:
    dfs: Dict[str, pd.DataFrame] = {}
    try:
        xls = pd.ExcelFile(io.BytesIO(uploaded.read()))
        for tab in xls.sheet_names:
            try:
                df = xls.parse(tab, dtype=str)
                dfs[tab] = df.fillna("")
            except Exception:
                pass
    except Exception as e:
        st.error(f"Failed reading Excel: {e}")
    return dfs

# ---------------- Route URL / ID extraction ----------------

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

def expand_scientific_to_digits(s: str) -> str:
    """
    Expand strings like '1.234E+12' into full integer digits as a string.
    If parsing fails, returns ''.
    """
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
        # e.g., 1234e+2 but frac longer than exp
        return digits  # fallback

def choose_best_route_id(url_val: str, source_id_val: str) -> str:
    """
    Try to get a Strava route id using either a URL cell or a Source ID cell.
    """
    # Prefer a valid URL if present
    rid = extract_route_id_from_url(url_val or "")
    if rid:
        return rid
    # Fall back to Source ID (which might be in scientific notation or similar)
    sid = str(source_id_val or "").strip()
    if not sid:
        return ""
    # If it's already digits, return it
    d = extract_digits(sid)
    if len(d) >= 6:
        return d
    # Else, try expanding sci notation
    exp = expand_scientific_to_digits(sid)
    if exp and exp.isdigit():
        return exp
    return ""

# ---------------- Bucketing ----------------

def normalize_group(v: str) -> Optional[str]:
    v = (v or "").strip().lower()
    if "5" in v:
        return "5k"
    if "8" in v:
        return "8k"
    return None

def normalize_surface(v: str) -> Optional[str]:
    v = (v or "").strip().lower()
    if "road" in v:
        return "Road"
    if "trail" in v or "trail" in v:
        return "Trail"
    # Try path/track hints
    if "path" in v or "off" in v:
        return "Trail"
    if "pavement" in v or "tarmac" in v:
        return "Road"
    return None

@dataclass
class BucketedRoute:
    route_id: str
    group: str   # '5k' or '8k'
    surface: str # 'Road' or 'Trail'

def bucket_routes(df: pd.DataFrame,
                  group_col: Optional[str],
                  surface_col: Optional[str],
                  url_col: Optional[str],
                  source_id_col: Optional[str]) -> Tuple[Dict[str, List[BucketedRoute]], List[str], pd.DataFrame]:
    """
    Returns:
      - buckets: dict with keys 'Road 5k','Trail 5k','Road 8k','Trail 8k' mapping to list of BucketedRoute
      - errors: list of error/warning strings
      - debug_rows: a small DataFrame of how each row was interpreted
    """
    buckets: Dict[str, List[BucketedRoute]] = {"Road 5k": [], "Trail 5k": [], "Road 8k": [], "Trail 8k": []}
    errors: List[str] = []
    dbg_records = []

    cols = {norm_header(c): c for c in df.columns}
    # Try to auto-detect if not provided
    if not group_col:
        for c in df.columns:
            nc = norm_header(c)
            if "group" in nc or "distance" in nc or "5k" in nc or "8k" in nc:
                group_col = c; break
    if not surface_col:
        for c in df.columns:
            nc = norm_header(c)
            if "surface" in nc or "road" in nc or "trail" in nc or "terrain" in nc:
                surface_col = c; break
    if not url_col:
        # pick a column that looks like it contains Strava URLs in at least one row
        for c in df.columns:
            if (df[c].astype(str).str.contains("strava.com", case=False, na=False)).any():
                url_col = c; break
        if not url_col:
            for c in df.columns:
                if "url" in norm_header(c) or "link" in norm_header(c) or "strava" in norm_header(c):
                    url_col = c; break
    if not source_id_col:
        for c in df.columns:
            nc = norm_header(c)
            if "sourceid" in nc or nc == "source" or nc.endswith("id"):
                source_id_col = c; break

    for i, row in df.iterrows():
        group_val = row.get(group_col, "") if group_col else ""
        surface_val = row.get(surface_col, "") if surface_col else ""
        url_val = row.get(url_col, "") if url_col else ""
        sid_val = row.get(source_id_col, "") if source_id_col else ""

        g = normalize_group(str(group_val))
        s = normalize_surface(str(surface_val))
        rid = choose_best_route_id(str(url_val), str(sid_val))

        dbg_records.append({
            "group_raw": group_val, "group_norm": g,
            "surface_raw": surface_val, "surface_norm": s,
            "url": url_val, "source_id": sid_val,
            "route_id": rid
        })

        if not (g and s and rid):
            continue
        key = f"{s} {g}"
        if key not in buckets:
            buckets[key] = []
        buckets[key].append(BucketedRoute(route_id=rid, group=g, surface=s))

    dbg_df = pd.DataFrame(dbg_records)
    # summary errors
    if dbg_df["route_id"].eq("").all():
        errors.append("No Strava route IDs could be extracted. Check the URL/Source ID mapping.")
    if dbg_df["group_norm"].isna().all():
        errors.append("Could not detect group (5k/8k). Map the correct 'Group/Distance' column.")
    if dbg_df["surface_norm"].isna().all():
        errors.append("Could not detect surface (Road/Trail). Map the correct 'Surface' column.")
    return buckets, errors, dbg_df

# ---------------- Strava download ----------------

def strava_export_gpx(route_id: str, token: str) -> bytes:
    url = f"https://www.strava.com/api/v3/routes/{route_id}/export_gpx"
    r = requests.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=30)
    if r.status_code == 429:
        # Rate limit: small backoff and one retry
        time.sleep(2)
        r = requests.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=30)
    r.raise_for_status()
    return r.content

# ---------------- Zipping ----------------

def build_bucket_zips(buckets: Dict[str, List[BucketedRoute]], token: str) -> Tuple[Dict[str, bytes], Dict[str, str]]:
    """
    Returns (zips_bytes, errors_by_route) where zips_bytes maps bucket name -> zip (bytes),
    and errors_by_route maps "bucket:route_id" -> error string.
    We dedupe downloads across all buckets.
    """
    # set of all unique route IDs
    all_ids: List[str] = []
    for blist in buckets.values():
        for br in blist:
            if br.route_id not in all_ids:
                all_ids.append(br.route_id)

    # download cache
    gpx_cache: Dict[str, Optional[bytes]] = {}
    errors_by_route: Dict[str, str] = {}

    # progress (if in Streamlit context)
    progress = st.progress(0.0)
    status = st.empty()
    total = len(all_ids)
    for idx, rid in enumerate(all_ids, start=1):
        status.write(f"Fetching GPX {idx}/{total} (route {rid}) …")
        try:
            gpx_cache[rid] = strava_export_gpx(rid, token)
        except Exception as e:
            gpx_cache[rid] = None
            errors_by_route[rid] = str(e)
        progress.progress(idx / max(total, 1))

    # Build one zip per bucket
    zips: Dict[str, bytes] = {}
    for bucket_name, blist in buckets.items():
        bio = io.BytesIO()
        with zipfile.ZipFile(bio, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for br in blist:
                g = gpx_cache.get(br.route_id)
                if not g:
                    errors_by_route[f"{bucket_name}:{br.route_id}"] = errors_by_route.get(br.route_id, "Unknown error")
                    continue
                # File name: use stable naming by ID
                zf.writestr(f"strava_route_{br.route_id}.gpx", g)
        zips[bucket_name] = bio.getvalue()
    status.write("Done.")
    return zips, errors_by_route
