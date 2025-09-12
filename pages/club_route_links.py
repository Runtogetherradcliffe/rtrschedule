# pages/club_route_links.py
# Build: v2025.08.15-STRAVA-IDFIX-9 (fix NameError in GPX export + progress; keep rate-limit handling)

import io
import re
import time
import urllib.parse
import pandas as pd
import requests
import streamlit as st

BUILD_ID = "v2025.08.15-STRAVA-IDFIX-9"

# --- Capture Strava token from ?code=... on redirect back to this page ---
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
        # remove code from URL
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

def source_id_to_long_digits(s: str) -> str:
    expanded = expand_sci_id(s)
    if expanded:
        return expanded
    return extract_digits(s)

def extract_route_id_from_url(u: str) -> str:
    m = STRAVA_ROUTE_ID_RE.search(u or "")
    return m.group(1) if m else ""

def extract_route_id(u: str, source_id: str) -> str:
    rid = extract_route_id_from_url(u)
    if rid:
        return rid
    return source_id_to_long_digits(source_id)

# --- URL-first reconciliation ---
def reconcile_route_url(url: str, source_id: str):
    dbg = {}
    rid_from_url = extract_route_id_from_url(url) if url else ""
    rid_from_src = source_id_to_long_digits(source_id)
    if rid_from_url:
        dbg["rid_from_url"] = rid_from_url
    if rid_from_src:
        dbg["rid_from_source"] = rid_from_src
    if (
        url and rid_from_url and rid_from_src
        and len(rid_from_url) < 12
        and len(rid_from_src) >= 12
    ):
        url = f"https://www.strava.com/routes/{rid_from_src}"
        dbg["url_rebuilt_from_source"] = url
    return url, dbg

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
        if r.status_code == 429:
            return "Rate-limited (429)"
        return f"API error {r.status_code}"
    except Exception as e:
        return f"API error: {e}"

# Simple in-session cache for route status to avoid re-hitting the API
if "route_api_cache" not in st.session_state:
    st.session_state["route_api_cache"] = {}

# ---------------------------- Load Data UI ----------------------------------
mode = st.radio("Load data from:", ["Google Sheet (CSV)", "Upload Excel (.xlsx)"], horizontal=True)

dfs = None
if mode.startswith("Google"):
url = "https://docs.google.com/spreadsheets/d/1ncT1NCbSnFsAokyFBkMWBVsk7yrJTiUfG0iBRxyUCTw/edit?usp=sharing"
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

        # URL-first: only synthesize from Source ID if URL is missing/placeholder
        url = make_https(url_raw)
        if (not url) and lt.lower().startswith("strava"):
            sid_digits = source_id_to_long_digits(sid_raw)
            if len(sid_digits) >= 12:
                url = f"https://www.strava.com/routes/{sid_digits}"

        rows_long.append(
            {"Date": d, "Route Name": nm, "Side": side, "Link Type": lt, "URL": url, "Source ID": sid_raw}
        )

links_df = pd.DataFrame(rows_long)
st.write(f"Found {len(links_df)} route link entries from the Schedule.")

debug = st.checkbox("Show debug for first 5 rows", value=True)

# ------------------------------- Validate -----------------------------------
st.subheader("Validate Links")
start_at = st.number_input("Start validating at row (0-based index)", min_value=0, max_value=max(0, len(links_df)-1), value=0, step=1)
limit = st.slider("Limit rows to validate in this pass", min_value=10, max_value=len(links_df), value=min(60, len(links_df)), step=10)
rate_delay = st.slider("Delay between API calls (seconds)", min_value=0.0, max_value=1.0, value=0.35, step=0.05, help="Increase if you hit 429s")

subset = links_df.iloc[start_at:start_at+limit].copy()
token = st.session_state.get("strava_token")
token_present = bool(token)

out_rows = []
hit_429 = False

for _, r in subset.iterrows():
    u = clean(r["URL"]); lt = clean(r["Link Type"]); sid_raw = clean(r["Source ID"])
    row_debug = {}
    if debug and len(out_rows) < 5:  # only annotate first 5 rows
        row_debug["raw_url"] = u or "(empty)"
        row_debug["link_type"] = lt
        row_debug["source_id_raw"] = sid_raw

    # URL-first reconcile: only rebuild if URL id is clearly short and source id is long
    if u and is_strava_route_url(u):
        u2, dbg2 = reconcile_route_url(u, sid_raw)
        if u2 != u:
            u = u2
        if debug and len(out_rows) < 5:
            row_debug.update(dbg2)

    # If still empty & Link Type is Strava, try synth (URL missing)
    if not u and lt.lower().startswith("strava"):
        sid_digits = source_id_to_long_digits(sid_raw)
        if len(sid_digits) >= 12:
            u = f"https://www.strava.com/routes/{sid_digits}"
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

        # cache check
        cache = st.session_state["route_api_cache"]
        cached = cache.get(rid)
        if cached:
            status = cached
        else:
            status = strava_route_status_via_api(rid, token)
            # cache OK and 404; also cache rate-limited this run to avoid hammering
            if status in ("OK (API)", "Not Found/No Access", "Unauthorized (token/scope)") or status.startswith("API error"):
                cache[rid] = status
            if status == "Rate-limited (429)":
                hit_429 = True

        out_rows.append({**r, "Status": status, "HTTP": "", "Content-Type": "", "Final URL": u, "Debug": row_debug if debug else ""})

        if hit_429:
            st.warning("Hit Strava rate limit (429). Stop this pass and resume later with a higher delay or smaller batch.")
            break

        if rate_delay > 0:
            time.sleep(rate_delay)

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
    if rate_delay > 0:
        time.sleep(rate_delay)

report_df = pd.DataFrame(out_rows).sort_values("Date")
st.dataframe(report_df, use_container_width=True, hide_index=True)
st.download_button(
    "⬇️ Download validation report (CSV)",
    data=report_df.to_csv(index=False).encode("utf-8"),
    file_name="route_link_validation.csv",
    mime="text/csv",
)

# ------------------------------- Export GPX ---------------------------------
st.subheader("Export GPX")

# Direct .gpx links
gpx_ok = report_df[(report_df["Status"].str.contains("OK")) & (report_df["URL"].str.lower().str.contains(".gpx"))]
if not gpx_ok.empty:
    pick = st.multiselect("Select direct-GPX routes to download", gpx_ok["Route Name"].tolist(), key="pick_gpx_direct")
    if st.button("Download selected direct GPX"):
        import zipfile
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for _, row in gpx_ok[gpx_ok["Route Name"].isin(pick)].iterrows():
                try:
                    resp = requests.get(row["URL"], timeout=20)
                    resp.raise_for_status()
                    base = urllib.parse.urlparse(row["URL"]).path.split("/")[-1] or "route.gpx"
                    zf.writestr(f"{row['Route Name'][:50].replace('/', '-')}-{base}", resp.content)
                except Exception as e:
                    st.warning(f"Failed to download {row['Route Name']}: {e}")
        st.download_button("⬇️ Download GPX ZIP (direct)", data=buf.getvalue(), file_name="routes_gpx.zip", mime="application/zip")

# Strava via API (works for your own public & private routes; others' public routes too)
token = st.session_state.get("strava_token")
if token:
    strava_routes = report_df[report_df["URL"].str.contains("strava.com") & report_df["URL"].str.contains("/routes/")]
    if not strava_routes.empty:
        st.markdown("#### Strava Routes via API")
        strava_routes = strava_routes.assign(**{"Route ID": strava_routes["URL"].str.extract(r"/routes/(\d+)").astype(str)})
        pick2 = st.multiselect("Select Strava Routes to export GPX", strava_routes["Route Name"].tolist(), key="pick_strava")
        export_delay = st.slider("Delay between Strava GPX downloads (seconds)", min_value=0.0, max_value=1.0, value=0.35, step=0.05)
        if st.button("Export selected Strava GPX") and pick2:
            import zipfile
            buf = io.BytesIO()
            prog = st.progress(0, text="Exporting GPX via Strava API...")
            total = len(pick2)
            done = 0
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for _, row in strava_routes[strava_routes["Route Name"].isin(pick2)].iterrows():
                    rid = row["Route ID"]
                    if not rid:
                        st.warning(f"Missing route id for {row['Route Name']}")
                        continue
                    api_url = f"https://www.strava.com/api/v3/routes/{rid}/export_gpx"
                    try:
                        resp = requests.get(api_url, headers={"Authorization": f"Bearer {token}"}, timeout=30)
                        if resp.status_code == 200:
                            zf.writestr(f"{row['Route Name'][:50].replace('/', '-')}-{rid}.gpx", resp.content)
                        elif resp.status_code == 401:
                            st.error("Unauthorized — token expired or missing scope. Reconnect on the Strava OAuth page.")
                            break
                        elif resp.status_code == 404:
                            st.warning(f"Route not found or not accessible: {row['Route Name']}")
                        elif resp.status_code == 429:
                            st.warning("Hit Strava rate limit while exporting. Try a smaller batch or increase delay.")
                            break
                        else:
                            st.warning(f"Strava API error {resp.status_code} for {row['Route Name']}")
                    except Exception as e:
                        st.warning(f"Failed to fetch GPX for {row['Route Name']}: {e}")
                    done += 1
                    prog.progress(min(done/total, 1.0))
                    if export_delay > 0:
                        time.sleep(export_delay)
            prog.empty()
            st.download_button("⬇️ Download Strava GPX ZIP", data=buf.getvalue(), file_name="routes_strava_gpx.zip", mime="application/zip")
else:
    st.info("Strava not connected. Open the Strava OAuth page and connect your account to export GPX for Strava routes.")
