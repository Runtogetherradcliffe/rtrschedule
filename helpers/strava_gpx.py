
# helpers/strava_gpx.py (v4 - friendly names from columns C/M, 'RTR - ' prefix)
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

# Expected letters for friendly names (user specified)
ROUTE1_NAME_LETTER = "C"  # Route 1 friendly name
ROUTE2_NAME_LETTER = "M"  # Route 2 friendly name

def _letter_to_index(letter: str) -> Optional[int]:
    letter = (letter or "").strip().upper()
    if not letter or not ("A" <= letter <= "Z"):
        return None
    return ord(letter) - ord("A")

def sanitize_filename(name: str) -> str:
    # Keep letters, numbers, spaces, dash, underscore, parentheses, dot
    name = re.sub(r"\s+", " ", str(name or "")).strip()
    if not name:
        return ""
    return re.sub(r"[^\w \-().]", "_", name)

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
    friendly: str  # friendly name for filename

ROUTE1_COLS = {
    "terrain": "Route 1 - Terrain (Road/Trail/Mixed)",
    "link_type": "Route 1 - Route Link Type",
    "url": "Route 1 - Route Link (Source URL)",
    "source_id": "Route 1 - Source ID",
    # optional header variants for friendly name
    "friendly_h1": "Route 1 - Friendly Name",
    "friendly_h2": "Route 1 - Route Name",
}
ROUTE2_COLS = {
    "terrain": "Route 2 - Terrain (Road/Trail/Mixed)",
    "link_type": "Route 2 - Route Link Type",
    "url": "Route 2 - Route Link (Source URL)",
    "source_id": "Route 2 - Source ID",
    "friendly_h1": "Route 2 - Friendly Name",
    "friendly_h2": "Route 2 - Route Name",
}

def norm_surface(v: str) -> Optional[str]:
    v = (v or "").strip().lower()
    if "road" in v:
        return "Road"
    if "trail" in v or "mixed" in v:
        return "Trail"  # Mixed -> Trail
    return None

def _friendly_from_letters(df_row, which: int, df_columns: list) -> str:
    # which=1 => column C, which=2 => column M
    if which == 1:
        idx = _letter_to_index(ROUTE1_NAME_LETTER)
    else:
        idx = _letter_to_index(ROUTE2_NAME_LETTER)
    if idx is not None and 0 <= idx < len(df_columns):
        try:
            return str(df_row[df_columns[idx]]).strip()
        except Exception:
            return ""
    return ""

def _friendly_from_headers(df_row, cols: dict) -> str:
    for k in ("friendly_h1", "friendly_h2"):
        h = cols.get(k)
        if h and h in df_row:
            v = str(df_row.get(h, "")).strip()
            if v:
                return v
    return ""

def add_row_to_bucket(row: dict, df_columns: list, cols: dict, fixed_group: str, buckets: Dict[str, List[BucketedRoute]]):
    link_type = str(row.get(cols["link_type"], "")).strip().lower()
    if link_type != "strava route":
        return
    rid = choose_route_id_from_cells(row.get(cols["url"], ""), row.get(cols["source_id"], ""))
    surf = norm_surface(row.get(cols["terrain"], ""))
    if not rid or not surf:
        return
    # Friendly name from header first, else by letter (C for Route1, M for Route2)
    friendly = _friendly_from_headers(row, cols)
    if not friendly:
        friendly = _friendly_from_letters(row, 1 if fixed_group=="8k" else 2, df_columns)  # Route1=8k -> C ; Route2=5k -> M
    key = f"{surf} {fixed_group}"
    buckets.setdefault(key, []).append(BucketedRoute(route_id=rid, group=fixed_group, surface=surf, friendly=friendly))

def build_buckets_from_master(df: pd.DataFrame) -> Tuple[Dict[str, List[BucketedRoute]], List[str]]:
    buckets: Dict[str, List[BucketedRoute]] = {
        "Road 5k": [], "Trail 5k": [], "Road 8k": [], "Trail 8k": []
    }
    errors: List[str] = []

    # Validate required columns exist
    required_cols = set(ROUTE1_COLS.values()) | set(ROUTE2_COLS.values())
    # friendly headers are optional, so remove them from the "required" set if missing in dict values
    required_cols.discard(ROUTE1_COLS["friendly_h1"])
    required_cols.discard(ROUTE1_COLS["friendly_h2"])
    required_cols.discard(ROUTE2_COLS["friendly_h1"])
    required_cols.discard(ROUTE2_COLS["friendly_h2"])

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        errors.append("Missing expected columns: " + ", ".join(missing))
        # We still continue, because friendly names can come from letters C/M and other columns are optional
        # but if critical link columns are missing, buckets will just be empty.

    df_columns = list(df.columns)
    for _, row in df.iterrows():
        # Route 1 -> 8k
        add_row_to_bucket(row, df_columns, ROUTE1_COLS, "8k", buckets)
        # Route 2 -> 5k
        add_row_to_bucket(row, df_columns, ROUTE2_COLS, "5k", buckets)

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

def _inject_gpx_name(gpx_bytes: bytes, friendly_name: str) -> bytes:
    """Ensure the GPX has a <name> set to friendly_name (prefixed with 'RTR - ')."""
    try:
        txt = gpx_bytes.decode("utf-8", errors="ignore")
    except Exception:
        return gpx_bytes
    safe_name = "RTR - " + (friendly_name or "").strip()
    if not safe_name.strip():
        return gpx_bytes
    # Replace first <name>...</name> if present
    if re.search(r"<name>.*?</name>", txt, flags=re.S):
        txt = re.sub(r"<name>.*?</name>", f"<name>{safe_name}</name>", txt, count=1, flags=re.S)
    else:
        # Insert after <gpx ...>
        txt = re.sub(r"(</metadata>)", f"\1\n  <name>{safe_name}</name>", txt, count=1) or txt
        if "<name>" not in txt:
            txt = re.sub(r"(\<gpx[^>]*\>)", f"\1\n  <name>{safe_name}</name>", txt, count=1) or txt
    return txt.encode("utf-8")

# ---------- Single zip with subfolders and friendly filenames ----------
def build_single_zip(buckets: Dict[str, List[BucketedRoute]], token: str):
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

    # Track used filenames per bucket to avoid collisions
    used_names_per_bucket: Dict[str, set] = {k: set() for k in buckets}

    bio = _io.BytesIO()
    with zipfile.ZipFile(bio, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # Create subfolders and add files
        for bucket_name, blist in buckets.items():
            zf.writestr(bucket_name.rstrip("/") + "/", b"")
            for br in blist:
                g = gpx_cache.get(br.route_id)
                if not g:
                    errors_by_route[f"{bucket_name}:{br.route_id}"] = errors_by_route.get(br.route_id, "Unknown error")
                    continue

                friendly = br.friendly.strip() if br.friendly else ""
                base = sanitize_filename(friendly) if friendly else ""
                if not base:
                    base = f"strava_route_{br.route_id}"
                final_name = f"RTR - {base}.gpx"
                # Avoid collisions inside each bucket
                if final_name in used_names_per_bucket[bucket_name]:
                    final_name = f"RTR - {base} ({br.route_id}).gpx"
                used_names_per_bucket[bucket_name].add(final_name)

                # Inject <name> into the GPX
                g2 = _inject_gpx_name(g, friendly or base)

                zf.writestr(f"{bucket_name}/{final_name}", g2)

        # Optional manifest
        manifest_lines = ["Buckets and counts:"]
        for k, v in buckets.items():
            manifest_lines.append(f"- {k}: {len(v)} routes")
        zf.writestr("MANIFEST.txt", "\n".join(manifest_lines))

    status.write("Done.")
    return bio.getvalue(), errors_by_route
