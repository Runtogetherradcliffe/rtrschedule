
import io
import re
import time
import urllib.parse
from dataclasses import dataclass
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Route Links & GPX", page_icon="🗺️", layout="wide")
st.title("🗺️ Route Links & GPX — Validate & Fetch (robust Strava checks)")

# -----------------------------
# Helpers
# -----------------------------
def clean(x):
    if pd.isna(x): return ""
    return str(x).strip()

def norm_header(h):
    return re.sub(r"[^a-z0-9]+", "", str(h).strip().lower())

def extract_sheet_id(url: str):
    m = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", url)
    return m.group(1) if m else None

def load_google_sheet_csv(sheet_id: str, sheet_name: str) -> pd.DataFrame:
    encoded = urllib.parse.quote(sheet_name, safe="")
    export_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={encoded}"
    df = pd.read_csv(export_url)
    if len(df.columns) and df.columns[0].lower().startswith("unnamed"):
        df = df.drop(columns=[df.columns[0]])
    return df

def load_from_google_csv(url: str):
    sheet_id = extract_sheet_id(url)
    dfs = {}
    # Schedule is the primary source
    df = load_google_sheet_csv(sheet_id, "Schedule")
    if not df.empty:
        dfs["Schedule"] = df
    # Optional
    for tab in ["Route Master", "RouteMaster", "Routemaster"]:
        try:
            df = load_google_sheet_csv(sheet_id, tab)
            if not df.empty:
                dfs["Route Master"] = df
                break
        except Exception:
            pass
    return dfs

def load_from_excel_bytes(bts: bytes):
    xls = pd.ExcelFile(io.BytesIO(bts))
    dfs = {}
    if "Schedule" in xls.sheet_names:
        dfs["Schedule"] = pd.read_excel(xls, "Schedule")
    for opt in ["Route Master", "RouteMaster"]:
        if opt in xls.sheet_names:
            dfs["Route Master"] = pd.read_excel(xls, opt)
            break
    return dfs

# -----------------------------
# Link parsing/normalization
# -----------------------------
@dataclass
class LinkInfo:
    link_type: str
    url: str
    source_id: str
    is_possible_gpx: bool

STRAVA_ROUTE_RE = re.compile(r"strava\.com/routes/(\d+)", re.I)
STRAVA_ACTIVITY_RE = re.compile(r"strava\.com/activities/(\d+)", re.I)
PLOTAROUTE_RE = re.compile(r"plotaroute\.com/route/(\d+)", re.I)
DIRECT_GPX_RE = re.compile(r"\.gpx(\?.*)?$", re.I)

def make_https(u: str) -> str:
    u = u.strip()
    if not u:
        return u
    parsed = urllib.parse.urlparse(u)
    if not parsed.scheme:
        u = "https://" + u
    return u

def normalize_url(link_type: str, url: str, source_id: str) -> str:
    lt = (link_type or "").strip().lower()
    url = (url or "").strip()
    sid = (source_id or "").strip()

    # If URL is just an ID, build a full URL
    if url.isdigit() and not sid:
        sid = url
        url = ""

    if lt in ["strava route", "strava"] or STRAVA_ROUTE_RE.search(url) or (lt.startswith("strava") and sid.isdigit()):
        if not url or url.isdigit():
            url = f"https://www.strava.com/routes/{sid}"
        return make_https(url)

    if lt in ["strava activity"] or STRAVA_ACTIVITY_RE.search(url) or ("activity" in lt and sid.isdigit()):
        if not url or url.isdigit():
            url = f"https://www.strava.com/activities/{sid}"
        return make_https(url)

    if lt in ["plotaroute"] or PLOTAROUTE_RE.search(url) or ("plotaroute" in lt and sid.isdigit()):
        if not url or url.isdigit():
            url = f"https://www.plotaroute.com/route/{sid}"
        return make_https(url)

    if lt in ["gpx"] or DIRECT_GPX_RE.search(url):
        return make_https(url)

    return make_https(url)

def parse_link(link_type: str, url: str) -> LinkInfo:
    lt = (link_type or "").strip().lower()
    url = (url or "").strip()
    sid = ""
    if lt in ["strava route", "strava"] or STRAVA_ROUTE_RE.search(url):
        m = STRAVA_ROUTE_RE.search(url)
        if m: sid = m.group(1)
        return LinkInfo("Strava Route", url, sid, False)
    if lt in ["strava activity"] or STRAVA_ACTIVITY_RE.search(url):
        m = STRAVA_ACTIVITY_RE.search(url)
        if m: sid = m.group(1)
        return LinkInfo("Strava Activity", url, sid, False)
    if lt in ["plotaroute"] or PLOTAROUTE_RE.search(url):
        m = PLOTAROUTE_RE.search(url)
        if m: sid = m.group(1)
        return LinkInfo("Plotaroute", url, sid, False)
    if lt in ["gpx"] or DIRECT_GPX_RE.search(url):
        return LinkInfo("GPX", url, "", True)
    return LinkInfo(link_type or "Unknown", url, "", DIRECT_GPX_RE.search(url) is not None)

# -----------------------------
# Robust HTTP check
# -----------------------------
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")

def check_url(u: str, timeout=15):
    sess = requests.Session()
    sess.headers.update({"User-Agent": UA, "Accept": "*/*"})
    result = {"ok": False, "status": None, "final_url": "", "content_type": "", "error": ""}

    try:
        # Try HEAD first
        r = sess.head(u, allow_redirects=True, timeout=timeout)
        result.update({"status": r.status_code, "final_url": r.url, "content_type": r.headers.get("Content-Type","")})
        if 200 <= r.status_code < 400 and result["content_type"]:
            result["ok"] = True
            return result
        # Fallback to GET (stream)
        r2 = sess.get(u, stream=True, allow_redirects=True, timeout=timeout)
        result.update({"status": r2.status_code, "final_url": r2.url, "content_type": r2.headers.get("Content-Type","")})
        if 200 <= r2.status_code < 400:
            result["ok"] = True
            r2.close()
            return result
        # Final fallback GET full
        r3 = sess.get(u, allow_redirects=True, timeout=timeout)
        result.update({"status": r3.status_code, "final_url": r3.url, "content_type": r3.headers.get("Content-Type","")})
        if 200 <= r3.status_code < 400:
            result["ok"] = True
        return result
    except Exception as e:
        result["error"] = str(e)
        return result

def classify_access(u: str, info: dict) -> str:
    if not info.get("ok"):
        if info.get("status") == 429:
            return "Rate-limited (try again)"
        return "Broken/Unreachable"
    final = (info.get("final_url") or "").lower()
    # Login redirects / private
    if "strava.com/login" in final or "strava.com/session" in final:
        return "Needs Auth/Not Public"
    if "private" in final:
        return "Not Public"
    # Content-type heuristics
    ct = (info.get("content_type") or "").lower()
    if "text/html" in ct or "text/plain" in ct or "application/xml" in ct or "application/gpx" in ct:
        return "OK"
    # default ok
    return "OK"

# -----------------------------
# Load UI
# -----------------------------
mode = st.radio("Load from:", ["Google Sheet (CSV)", "Upload Excel (.xlsx)"], horizontal=True)

dfs = None
if mode == "Google Sheet (CSV)":
    gs_url = st.text_input("Google Sheet URL")
    if gs_url:
        try:
            dfs = load_from_google_csv(gs_url)
        except Exception as e:
            st.error(f"Could not read Google Sheet: {e}")
elif mode == "Upload Excel (.xlsx)":
    up = st.file_uploader("Upload master Excel (.xlsx)", type=["xlsx"])
    if up is not None:
        try:
            dfs = load_from_excel_bytes(up.read())
        except Exception as e:
            st.error(f"Could not read Excel: {e}")

if not dfs or "Schedule" not in dfs:
    st.error("Could not load a 'Schedule' tab. Please verify.")
    st.stop()

sched = dfs["Schedule"]

# Build links from Schedule
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
r_urls  = [find_col(["route1routelinksourceurl", "route1routelink", "route1url"]), find_col(["route2routelinksourceurl", "route2routelink", "route2url"])]
r_srcid = [find_col(["route1sourceid", "route1id"]), find_col(["route2sourceid", "route2id"])]

if not date_col or not all(r_names) or not all(r_types) or not all(r_urls):
    st.error("Schedule is missing expected columns for route names/link types/URLs.")
    st.stop()

long_rows = []
for _, row in sched.iterrows():
    d = row[date_col]
    for i, side in enumerate(["1","2"]):
        nm = clean(row[r_names[i]])
        lt = clean(row[r_types[i]])
        url = clean(row[r_urls[i]])
        sid = clean(row[r_srcid[i]]) if r_srcid[i] else ""
        if not nm or nm.lower() == "no run":
            continue
        url_norm = normalize_url(lt, url, sid)
        long_rows.append({"Date": d, "Route Name": nm, "Side": side, "Link Type": lt, "URL": url_norm, "Source ID": sid})

links_df = pd.DataFrame(long_rows)
if links_df.empty:
    st.warning("No route links found in the Schedule."); st.stop()

st.write(f"Found {len(links_df)} route link entries from the Schedule.")

# Validate links (with UA + fallbacks)
st.subheader("Validate Links")
sample_limit = st.slider("Limit rows to validate", min_value=10, max_value=links_df.shape[0], value=min(200, links_df.shape[0]), step=10)

subset = links_df.head(sample_limit).copy()
rows = []
for _, r in subset.iterrows():
    name = clean(r["Route Name"]); url = clean(r["URL"]); ltype = clean(r["Link Type"])
    if not url:
        rows.append({"Date": r["Date"], "Route Name": name, "Side": r["Side"], "Link Type": ltype or "Unknown", "URL": "", "Status": "Missing URL", "HTTP": "", "Content-Type": "", "Final URL": "", "Source ID": r["Source ID"]})
        continue
    info = check_url(url)
    status = classify_access(url, info)
    rows.append({
        "Date": r["Date"],
        "Route Name": name,
        "Side": r["Side"],
        "Link Type": ltype or "Unknown",
        "URL": url,
        "Status": status,
        "HTTP": info.get("status"),
        "Content-Type": info.get("content_type"),
        "Final URL": info.get("final_url"),
        "Source ID": r["Source ID"]
    })
    # be polite to servers
    time.sleep(0.2)

report_df = pd.DataFrame(rows).sort_values("Date")
st.dataframe(report_df, use_container_width=True, hide_index=True)
st.download_button("⬇️ Download validation report (CSV)", data=report_df.to_csv(index=False).encode("utf-8"), file_name="route_link_validation.csv", mime="text/csv")

# GPX download for direct links
st.subheader("Fetch GPX (direct .gpx links only)")
gpx_candidates = report_df[(report_df["Status"] == "OK") & (report_df["URL"].str.lower().str.contains(".gpx"))]
if gpx_candidates.empty:
    st.caption("No direct .gpx links detected in the validated rows. Public Strava/Plotaroute can be validated; GPX export for Strava pages needs OAuth unless you have direct GPX URLs.")
else:
    pick = st.multiselect("Select routes to download GPX", gpx_candidates["Route Name"].tolist())
    if st.button("Download selected GPX"):
        zip_buf = io.BytesIO()
        import zipfile
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for _, row in gpx_candidates[gpx_candidates["Route Name"].isin(pick)].iterrows():
                url = row["URL"]
                try:
                    resp = requests.get(url, timeout=20); resp.raise_for_status()
                    base = urllib.parse.urlparse(url).path.split("/")[-1] or "route.gpx"
                    fname = f"{row['Route Name'][:50].replace('/', '-')}-{base}"
                    zf.writestr(fname, resp.content)
                except Exception as e:
                    st.warning(f"Failed to download {row['Route Name']}: {e}")
        st.download_button("⬇️ Download GPX ZIP", data=zip_buf.getvalue(), file_name="routes_gpx.zip", mime="application/zip")
