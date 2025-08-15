# pages/club_route_links.py
# Build: v2025.08.15-STRAVA-IDFIX-3

import io
import re
import time
import urllib.parse
import pandas as pd
import requests
import streamlit as st

BUILD_ID = "v2025.08.15-STRAVA-IDFIX-3"

# --- Capture Strava token from ?code=... on any redirect back to this page ---
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

st.set_page_config(page_title="Route Links & GPX", page_icon="🗺️", layout="wide")
st.title("🗺️ Route Links & GPX — Validate & Fetch (Strava-aware)")
st.caption(f"Build: {BUILD_ID}")

# Capture token if redirected here after OAuth
capture_strava_token_from_query()

# ----------------------------- Utilities ------------------------------------
def clean(x):
    return "" if pd.isna(x) else str(x).strip()

def norm_header(h):
    return re.sub(r"[^a-z0-9]+", "", str(h).strip().lower())

def extract_sheet_id(url):
    m = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", url)
    return m.group(1) if m else None

def load_google_sheet_csv(sheet_id, sheet_name):
    # dtype=str preserves big numeric IDs; keep_default_na=False stops "NA" -> NaN
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

def make_https(u):
    """Normalize URL; treat placeholders like 'Strava Route' (or any with spaces) as empty."""
    u = (u or "").strip()
    if not u:
        return u
    if " " in u or u.lower().startswith("strava route"):
        return ""  # placeholder/invalid
    if not urllib.parse.urlparse(u).scheme:
        return "https://" + u
    return u

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

def try_get_public(u, timeout=15):
    s = requests.Session()
    s.headers.update({"User-Agent": UA, "Accept": "*/*"})
    out = {"ok": False, "status": None, "final_url": "", "content_type": "", "error": ""}
    try:
        r = s.get(u, stream=True, allow_redirects=True, timeout=timeout)
        out.update(
            {"status": r.status_code, "final_url": r.url, "content_type": r.headers.get("Content-Type", "")}
        )
        out["ok"] = 200 <= r.status_code < 400
        r.close()
    except Exception as e:
        out["error"] = str(e)
    return out

# --- Strava helpers ---
STRAVA_ROUTE_ID_RE = re.compile(r"(?:^|/)(?:routes|routes/view)/(\d+)(?:[/?#].*)?$", re.I)

def is_strava_route_url(u: str) -> bool:
    if not u:
        return False
    lu = u.lower()
    return ("strava.com" in lu) and ("/routes/" in lu)

def extract_digits(s: str) -> str:
    # Pulls all digits; works for "3.3357e+18" or "3,335,7..." etc.
    return "".join(re.findall(r"\d+", s or ""))

def extract_route_id(u: str, source_id: str) -> str:
    """Prefer ID from URL; else digits from Source ID."""
    m = STRAVA_ROUTE_ID_RE.search(u or "")
    if m:
        return m.group(1)
    sid = extract_digits(source_id)
    return sid

def strava_route_status_via_api(route_id: str, token: str) -> str:
    if not route_id:
        return "Missing route id"
    try:
        r = requests.get(
            f"https://www.strava.com/api/v3/routes/{route_id}",
            headers={"Authorization": f"Bearer {token}"},
            timeout=15,
        )
        if r.status_code == 200:
            return "OK (API)"
        if r.status_code == 401:
            return "Unauthorized (token/scope)"
        if r.status_code == 404:
            return "Not Found/No Access"
        return f"API error {r.status_code}"
    except Exception as e:
        return f"API error: {e}"

# ---------------------------- Load Data UI ----------------------------------
mode = st.radio("Load data from:", ["Google Sheet (CSV)", "Upload Excel (.xlsx)"], horizontal=True)

dfs = None
if mode.startswith("Google"):
    url = st.text_input("Google Sheet URL")
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
sched.columns = [str(c) for c in sched.columns]  # ensure str headers
cols = {c: norm_header(c) for c in sched.columns}

def find_col(targets):
    for c, n in cols.items():
        for t in targets:
            if t in n:
                return c
    return None

date_col = find_col(["date", "datethu"])
r_names = [find_col(["route1name"]), find_col(["route2name"])]
r_types = [find_col(["route1routelinktype", "route1linktype"]), find_col(["route2routelinktype", "route2linktype"])]
r_urls  = [find_col(["route1routelinksourceurl", "route1routelink", "route1url"]),
           find_col(["route2routelinksourceurl", "route2routelink", "route2url"])]
r_srcid = [find_col(["route1sourceid", "route1id"]), find_col(["route2sourceid", "route2id"])]

if not date_col or not all(r_names) or not all(r_types) or not all(r_urls):
    st.error("Schedule is missing expected columns for route names/link types/URLs.")
    st.stop()

# -------------------------- Normalize rows (long) ---------------------------
rows_long = []
for _, row in sched.iterrows():
    d = row.get(date_col, "")
    for i, side in enumerate(["1", "2"]):
        nm = clean(row.get(r_names[i], ""))
        lt = clean(row.get(r_types[i], ""))
        url_raw = clean(row.get(r_urls[i], ""))
        sid_raw = clean(row.get(r_srcid[i], "")) if r_srcid[i] else ""
        if not nm or nm.lower() == "no run":
            continue
        url = make_https(url_raw)
        # If URL is missing/placeholder but Link Type says Strava and Source ID has digits => synth canonical URL
        if (not url) and lt.lower().startswith("strava"):
            rid_guess = extract_digits(sid_raw)
            if len(rid_guess) >= 6:
                url = f"https://www.strava.com/routes/{rid_guess}"
        rows_long.append(
            {"Date": d, "Route Name": nm, "Side": side, "Link Type": lt, "URL": url, "Source ID": sid_raw}
        )

links_df = pd.DataFrame(rows_long)
st.write(f"Found {len(links_df)} route link entries from the Schedule.")

debug = st.checkbox("Show debug for first 5 rows", value=True)

# ------------------------------- Validate -----------------------------------
st.subheader("Validate Links")
limit = st.slider(
    "Limit rows to validate", min_value=10, max_value=len(links_df), value=min(200, len(links_df)), step=10
)

subset = links_df.head(limit).copy()
token = st.session_state.get("strava_token")
token_present = bool(token)

out_rows = []
for idx, r in subset.iterrows():
    u = clean(r["URL"]); lt = clean(r["Link Type"]); sid_raw = clean(r["Source ID"])
    row_debug = {}
    if debug and len(out_rows) < 5:  # only annotate first 5 rows
        row_debug["raw_url"] = u or "(empty)"
        row_debug["link_type"] = lt
        row_debug["source_id_raw"] = sid_raw

    # If still empty & Link Type is Strava, try synth again
    if not u and lt.lower().startswith("strava"):
        rid_guess = extract_digits(sid_raw)
        if len(rid_guess) >= 6:
            u = f"https://www.strava.com/routes/{rid_guess}"
            if debug and len(out_rows) < 5:
                row_debug["synth_url_from_id"] = u

    if not u:
        out_rows.append({**r, "Status": "Missing URL", "HTTP": "", "Content-Type": "", "Final URL": "", "Debug": row_debug if debug else ""})
        continue

    # Prefer Strava API path if connected and URL looks like a Strava route
    if token_present and is_strava_route_url(u):
        rid = extract_route_id(u, sid_raw)
        if debug and len(out_rows) < 5:
            row_debug["route_id_extracted"] = rid
        api_status = strava_route_status_via_api(rid, token) if rid else "Missing route id"
        out_rows.append({**r, "Status": api_status, "HTTP": "", "Content-Type": "", "Final URL": u, "Debug": row_debug if debug else ""})
        time.sleep(0.05)
        continue

    # Fallback: public fetch (non-Strava or not connected)
    info = try_get_public(u)
    final_status = (
        "Needs Auth/Not Public" if ("strava.com" in u.lower() and "/routes/" in u.lower() and not token_present)
        else ("Rate-limited (try again)" if info.get("status") == 429 else ("OK" if info.get("ok") else "Broken/Unreachable"))
    )
    if debug and len(out_rows) < 5:
        row_debug["public_fetch"] = {"status": info.get("status"), "final_url": info.get("final_url")}
    out_rows.append({
        **r,
        "Status": final_status,
        "HTTP": info.get("status"),
        "Content-Type": info.get("content_type"),
        "Final URL": info.get("final_url"),
        "Debug": row_debug if debug else ""
    })
    time.sleep(0.05)

report_df = pd.DataFrame(out_rows).sort_values("Date")
st.dataframe(report_df, use_container_width=True, hide_index=True)
st.download_button(
    "⬇️ Download validation report (CSV)",
    data=report_df.to_csv(index=False).encode("utf-8"),
    file_name="route_link_validation.csv",
    mime="text/csv",
)
