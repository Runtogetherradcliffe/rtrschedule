
import io, re, time, urllib.parse, pandas as pd, requests, streamlit as st

import streamlit as st, requests

def capture_strava_token_from_query():
    """If the URL contains ?code=... from Strava and we don't yet have a token,
    exchange it here so any page can complete OAuth."""
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
        st.error("Strava token exchange failed: {}".format(e))

st.set_page_config(page_title="Route Links & GPX", page_icon="🗺️", layout="wide")
st.title("🗺️ Route Links & GPX — Validate & Fetch (Strava-aware)")

# Capture token if redirected here after OAuth
capture_strava_token_from_query()

def clean(x):
    return "" if pd.isna(x) else str(x).strip()

def norm_header(h):
    return re.sub(r"[^a-z0-9]+","",str(h).strip().lower())

def extract_sheet_id(url):
    m = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", url)
    return m.group(1) if m else None

def load_google_sheet_csv(sheet_id, sheet_name):
    u = "https://docs.google.com/spreadsheets/d/{}/gviz/tq?tqx=out:csv&sheet=".format(sheet_id) + urllib.parse.quote(sheet_name, safe="")
    df = pd.read_csv(u)
    if len(df.columns) and str(df.columns[0]).lower().startswith("unnamed"):
        df = df.drop(columns=[df.columns[0]])
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
        dfs["Schedule"] = pd.read_excel(xls, "Schedule")
    return dfs

def make_https(u):
    u = (u or "").strip()
    if not u:
        return u
    if not urllib.parse.urlparse(u).scheme:
        return "https://" + u
    return u

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")

def try_get_public(u, timeout=15):
    s = requests.Session()
    s.headers.update({"User-Agent": UA, "Accept": "*/*"})
    out = {"ok": False, "status": None, "final_url": "", "content_type": "", "error": ""}
    try:
        r = s.get(u, stream=True, allow_redirects=True, timeout=timeout)
        out.update({"status": r.status_code, "final_url": r.url, "content_type": r.headers.get("Content-Type","")})
        out["ok"] = (200 <= r.status_code < 400)
        r.close()
        return out
    except Exception as e:
        out["error"] = str(e)
        return out

def classify(u, info, token_present):
    if "strava.com/routes/" in u.lower():
        if not token_present:
            return "Needs Auth/Not Public"
    if not info.get("ok"):
        return "Rate-limited (try again)" if info.get("status") == 429 else "Broken/Unreachable"
    final = (info.get("final_url") or "").lower()
    if "strava.com/login" in final or "strava.com/session" in final:
        return "Needs Auth/Not Public"
    return "OK"

mode = st.radio("Load from:", ["Google Sheet (CSV)", "Upload Excel (.xlsx)"], horizontal=True)

dfs = None
if mode.startswith("Google"):
    url = st.text_input("Google Sheet URL")
    if url:
        try:
            dfs = load_from_google_csv(url)
        except Exception as e:
            st.error("Could not read Google Sheet: {}".format(e))
else:
    up = st.file_uploader("Upload master Excel (.xlsx)", type=["xlsx"])
    if up:
        try:
            dfs = load_from_excel_bytes(up.read())
        except Exception as e:
            st.error("Could not read Excel: {}".format(e))

if not dfs or "Schedule" not in dfs:
    st.error("Could not load a 'Schedule' tab.")
    st.stop()

sched = dfs["Schedule"]
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

rows_long = []
for _, row in sched.iterrows():
    d = row[date_col]
    for i, side in enumerate(["1","2"]):
        nm = clean(row[r_names[i]])
        lt = clean(row[r_types[i]])
        url = clean(row[r_urls[i]])
        sid = clean(row[r_srcid[i]]) if r_srcid[i] else ""
        if not nm or nm.lower() == "no run":
            continue
        if (lt.lower().startswith("strava") and sid.isdigit() and (not url or url.isdigit())):
            url = "https://www.strava.com/routes/{}".format(sid)
        url = make_https(url)
        rows_long.append({"Date": d, "Route Name": nm, "Side": side, "Link Type": lt, "URL": url, "Source ID": sid})

links_df = pd.DataFrame(rows_long)
st.write("Found {} route link entries from the Schedule.".format(len(links_df)))

st.subheader("Validate Links")
limit = st.slider("Limit rows to validate", min_value=10, max_value=len(links_df), value=min(200, len(links_df)), step=10)

subset = links_df.head(limit).copy()
token_present = bool(st.session_state.get("strava_token"))
out_rows = []
for _, r in subset.iterrows():
    u = clean(r["URL"])
    if not u:
        out_rows.append({**r, "Status":"Missing URL","HTTP":"","Content-Type":"","Final URL":""})
        continue
    info = try_get_public(u)
    status = classify(u, info, token_present)
    out_rows.append({**r, "Status":status,"HTTP":info.get("status"),"Content-Type":info.get("content_type"),"Final URL":info.get("final_url")})
    time.sleep(0.1)

report_df = pd.DataFrame(out_rows).sort_values("Date")
st.dataframe(report_df, use_container_width=True, hide_index=True)
st.download_button("⬇️ Download validation report (CSV)", data=report_df.to_csv(index=False).encode("utf-8"), file_name="route_link_validation.csv", mime="text/csv")

st.subheader("Export GPX")
token = st.session_state.get("strava_token")

gpx_ok = report_df[(report_df["Status"].str.contains("OK")) & (report_df["URL"].str.lower().str.contains(".gpx"))]
if not gpx_ok.empty:
    pick = st.multiselect("Select direct-GPX routes to download", gpx_ok["Route Name"].tolist(), key="pick_gpx_direct")
    if st.button("Download selected direct GPX"):
        import zipfile, io
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for _, row in gpx_ok[gpx_ok["Route Name"].isin(pick)].iterrows():
                try:
                    resp = requests.get(row["URL"], timeout=20)
                    resp.raise_for_status()
                    base = urllib.parse.urlparse(row["URL"]).path.split("/")[-1] or "route.gpx"
                    zf.writestr("{}-{}".format(row["Route Name"][:50].replace("/","-"), base), resp.content)
                except Exception as e:
                    st.warning("Failed to download {}: {}".format(row["Route Name"], e))
        st.download_button("⬇️ Download GPX ZIP (direct)", data=buf.getvalue(), file_name="routes_gpx.zip", mime="application/zip")

if token:
    strava_routes = report_df[report_df["URL"].str.contains("strava.com/routes/")]
    if not strava_routes.empty:
        st.markdown("#### Strava Routes via API")
        strava_routes = strava_routes.assign(**{"Route ID": strava_routes["URL"].str.extract(r"/routes/(\d+)").astype(str)})
        pick2 = st.multiselect("Select Strava Routes to export GPX", strava_routes["Route Name"].tolist(), key="pick_strava")
        if st.button("Export selected Strava GPX") and pick2:
            import zipfile, io
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for _, row in strava_routes[strava_routes["Route Name"].isin(pick2)].iterrows():
                    rid = row["Route ID"]
                    if not rid:
                        st.warning("Missing route id for {}".format(row["Route Name"]))
                        continue
                    api_url = "https://www.strava.com/api/v3/routes/{}/export_gpx".format(rid)
                    try:
                        resp = requests.get(api_url, headers={"Authorization": "Bearer {}".format(token)}, timeout=20)
                        if resp.status_code == 200:
                            zf.writestr("{}-{}.gpx".format(row["Route Name"][:50].replace("/","-"), rid), resp.content)
                        elif resp.status_code == 401:
                            st.error("Unauthorized — token expired or missing scope. Reconnect on the Strava OAuth page.")
                            break
                        elif resp.status_code == 404:
                            st.warning("Route not found or not accessible: {}".format(row["Route Name"]))
                        else:
                            st.warning("Strava API error {} for {}".format(resp.status_code, row["Route Name"]))
                    except Exception as e:
                        st.warning("Failed to fetch GPX for {}: {}".format(row["Route Name"], e))
            st.download_button("⬇️ Download Strava GPX ZIP", data=buf.getvalue(), file_name="routes_strava_gpx.zip", mime="application/zip")
else:
    st.info("Strava not connected. Open the Strava OAuth page and connect your account to export GPX for Strava routes.")
