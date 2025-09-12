# pages/route_metadata.py
# Build: v2025.08.15-METADATA-3 (synthesize URL from Source ID when blank)

import io
import re
import time
import urllib.parse
import pandas as pd
import requests
import streamlit as st

BUILD_ID = "v2025.08.15-METADATA-3"

# ----------------------------- OAuth capture ------------------------------
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
        # Remove ?code from URL so refreshes don't re-exchange
        qp = dict(st.query_params)
        if "code" in qp:
            del qp["code"]
            st.query_params.clear()
            for k, v in qp.items():
                st.query_params[k] = v
        st.success("Strava connected")
    except Exception as e:
        st.error(f"Strava token exchange failed: {e}")

st.set_page_config(page_title="Route Metadata (Strava)", page_icon="📏", layout="wide")
st.title("📏 Route Metadata — Strava Distance & Elevation")
st.caption(f"Build: {BUILD_ID}")

capture_strava_token_from_query()

# ----------------------------- Utilities ------------------------------------
def clean(x):
    return "" if pd.isna(x) else str(x).strip()

def norm_header(h):
    import re as _re
    return _re.sub(r"[^a-z0-9]+", "", str(h).strip().lower())

def extract_sheet_id(url):
    import re as _re
    m = _re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", url)
    return m.group(1) if m else None

def load_google_sheet_csv(sheet_id, sheet_name):
    u = (
        f"https://docs.google.com/spreadsheets/d/{sheet_id}"
        "/gviz/tq?tqx=out:csv&sheet=" + urllib.parse.quote(sheet_name, safe="")
    )
    df = pd.read_csv(u, dtype=str, keep_default_na=False)
    return df

def load_from_google_csv(url):
    sid = extract_sheet_id(url)
    dfs = {}
    if sid:
        # Only the Schedule tab is required
        df = load_google_sheet_csv(sid, "Schedule")
        if not df.empty:
            dfs["Schedule"] = df
    return dfs

def load_from_excel_bytes(bts):
    xls = pd.ExcelFile(io.BytesIO(bts))
    dfs = {}
    if "Schedule" in xls.sheet_names:
        dfs["Schedule"] = pd.read_excel(xls, "Schedule", dtype=str)
    return dfs

# URL helpers
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

def extract_route_id(u: str, source_id: str) -> str:
    rid = extract_route_id_from_url(u)
    if rid:
        return rid
    return source_id_to_digits(source_id)

# API helpers
def strava_get_route(route_id: str, token: str):
    url = f"https://www.strava.com/api/v3/routes/{route_id}"
    r = requests.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=20)
    return r

def parse_route_metadata(route_json: dict) -> dict:
    dist_m = route_json.get("distance")
    elev_m = route_json.get("elevation_gain") or route_json.get("elevation_gain_total") or route_json.get("elevation")
    name = route_json.get("name")
    sport_type = route_json.get("type") or route_json.get("sport_type")
    dist_km = None
    try:
        if dist_m is not None:
            dist_km = round(float(dist_m) / 1000.0, 2)
    except Exception:
        dist_km = None
    elev_m_val = None
    try:
        if elev_m is not None:
            elev_m_val = round(float(elev_m), 1)
    except Exception:
        elev_m_val = None
    return {
        "Route Name (API)": name,
        "Distance (km)": dist_km,
        "Elevation Gain (m)": elev_m_val,
        "Sport/Type": sport_type,
    }

# in-session caches
if "route_json_cache" not in st.session_state:
    st.session_state["route_json_cache"] = {}

# ---------------------------- Load Data UI ----------------------------------
mode = st.radio("Load data from:", ["Google Sheet (CSV)", "Upload Excel (.xlsx)"], horizontal=True)

dfs = None
if mode.startswith("Google"):
    url = ("https://docs.google.com/spreadsheets/d/1ncT1NCbSnFsAokyFBkMWBVsk7yrJTiUfG0iBRxyUCTw/edit?usp=sharing") if False else "https://docs.google.com/spreadsheets/d/1ncT1NCbSnFsAokyFBkMWBVsk7yrJTiUfG0iBRxyUCTw/edit?usp=sharing"
    if url:
        try:
            dfs = load_from_google_csv(url)
        except Exception as e:
            st.error(f"Could not read Google Sheet: {e}")
else:
    up = st.file_uploader("Upload master Excel (.xlsx)", type=["xlsx"])
    if up:
        try:
            dfs = load_from_excel_bytes(up.read())
        except Exception as e:
            st.error(f"Could not read Excel: {e}")

if not dfs or "Schedule" not in dfs:
    st.error("Could not load a 'Schedule' tab.")
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

if not date_col or not all(r_names) or not all(r_types) or not all(r_urls):
    st.error("Schedule is missing expected columns for route names/link types/URLs.")
    st.stop()

# -------------------------- Flatten rows (long) -----------------------------
synth_count = 0
rows_long = []
for _, row in sched.iterrows():
    d = row.get(date_col, "")
    for i, side in enumerate(["1","2"]):
        nm = clean(row.get(r_names[i], ""))
        lt = clean(row.get(r_types[i], ""))
        url_raw = clean(row.get(r_urls[i], ""))
        sid_raw = clean(row.get(r_srcid[i], "")) if r_srcid[i] else ""
        if not nm or nm.lower() == "no run":
            continue
        url = make_https(url_raw)

        # NEW: synthesize URL from Source ID when URL missing and Link Type mentions Strava
        if (not url) and ("strava" in lt.lower() or "strava" in nm.lower()):
            sid_digits = source_id_to_digits(sid_raw)
            # accept both long and older short route IDs (>= 6 digits)
            if len(sid_digits) >= 6:
                url = f"https://www.strava.com/routes/{sid_digits}"
                synth_count += 1

        rid = extract_route_id(url, sid_raw) if ("strava" in lt.lower() or is_strava_route_url(url)) else ""
        rows_long.append({"Date": d, "Route Name": nm, "Side": side, "Link Type": lt, "URL": url, "Route ID": rid})

links_df = pd.DataFrame(rows_long)
st.write(f"Found {len(links_df)} route entries. (Synthesized {synth_count} Strava URLs from Source IDs)")

# ---------------------------- Validation & Metadata -------------------------
st.subheader("Validate & Fetch Metadata")
token = st.session_state.get("strava_token")
if not token:
    st.info("Connect your Strava account on the OAuth page first (needs read/read_all).")

start_at = st.number_input("Start at row", min_value=0, max_value=max(0, len(links_df)-1), value=0, step=1)
limit = st.slider("Rows in this pass", min_value=10, max_value=len(links_df), value=min(60, len(links_df)), step=10)
delay = st.slider("Delay between API calls (seconds)", min_value=0.0, max_value=1.0, value=0.3, step=0.05)

subset = links_df.iloc[start_at:start_at+limit].copy()

def strava_get_route(route_id: str, token: str):
    url = f"https://www.strava.com/api/v3/routes/{route_id}"
    r = requests.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=20)
    return r

def parse_route_metadata(route_json: dict) -> dict:
    dist_m = route_json.get("distance")
    elev_m = route_json.get("elevation_gain") or route_json.get("elevation_gain_total") or route_json.get("elevation")
    name = route_json.get("name")
    sport_type = route_json.get("type") or route_json.get("sport_type")
    dist_km = None
    try:
        if dist_m is not None:
            dist_km = round(float(dist_m) / 1000.0, 2)
    except Exception:
        dist_km = None
    elev_m_val = None
    try:
        if elev_m is not None:
            elev_m_val = round(float(elev_m), 1)
    except Exception:
        elev_m_val = None
    return {
        "Route Name (API)": name,
        "Distance (km)": dist_km,
        "Elevation Gain (m)": elev_m_val,
        "Sport/Type": sport_type,
    }

out_rows = []
hit_429 = False
prog = st.progress(0, text="Working...")

for i, (_, r) in enumerate(subset.iterrows(), start=1):
    url = r["URL"]; lt = r["Link Type"]; rid = r["Route ID"]
    status = None
    dist_km = None; elev_m = None; name_api = None; sport_type = None

    if not url and "strava" not in lt.lower():
        status = "Non-Strava link (no URL)"
    elif url and not is_strava_route_url(url):
        status = "Non-Strava link (URL)"
    elif not rid:
        status = "Missing route id (URL/Source ID)"
    elif not token:
        status = "Needs Strava login"
    else:
        cache = st.session_state["route_json_cache"]
        if rid in cache:
            data = cache[rid]
            status = "OK (cached)"
        else:
            resp = strava_get_route(rid, token)
            if resp.status_code == 200:
                data = resp.json()
                cache[rid] = data
                status = "OK (API)"
            elif resp.status_code == 404:
                data = None; status = "Not Found/No Access"
            elif resp.status_code == 401:
                data = None; status = "Unauthorized (token/scope)"
            elif resp.status_code == 429:
                data = None; status = "Rate-limited (429)"; hit_429 = True
            else:
                data = None; status = f"API error {resp.status_code}"

        if data:
            parsed = parse_route_metadata(data)
            name_api = parsed["Route Name (API)"]
            dist_km = parsed["Distance (km)"]
            elev_m = parsed["Elevation Gain (m)"]
            sport_type = parsed["Sport/Type"]
        if delay > 0 and status.startswith("OK (API)"):
            time.sleep(delay)

    out_rows.append({
        "Date": r["Date"],
        "Route Name": r["Route Name"],
        "Side": r["Side"],
        "URL": url,
        "Route ID": rid,
        "Status": status or "Skipped",
        "Distance (km)": dist_km,
        "Elevation Gain (m)": elev_m,
        "Sport/Type": sport_type,
    })

    prog.progress(min(i/len(subset), 1.0))

    if hit_429:
        st.warning("Hit Strava rate limit (429). Stop this pass and resume later with a higher delay or smaller batch.")
        break

prog.empty()

meta_df = pd.DataFrame(out_rows)
st.dataframe(meta_df, use_container_width=True, hide_index=True)

st.download_button(
    "⬇️ Download route metadata (CSV)",
    data=meta_df.to_csv(index=False).encode("utf-8"),
    file_name="route_metadata.csv",
    mime="text/csv",
)

st.info("Tip: keep Source ID columns as Plain text in Sheets. This page now synthesizes Strava URLs from Source IDs when URL cells are blank.")
