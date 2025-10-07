
# pages/download_gpx_zips.py
# Build: v2025.10.07-GPXZIP-1

import streamlit as st
import pandas as pd
import urllib.parse
from helpers.strava_gpx import (
    capture_strava_token_from_query,
    load_from_google_csv,
    load_from_uploaded_excel,
    bucket_routes,
    build_bucket_zips,
)

st.set_page_config(page_title="Download GPX Zips (Strava)", page_icon="📦", layout="wide")
st.title("📦 Download GPX Zips — Strava routes from Schedule")
st.caption("Build: v2025.10.07-GPXZIP-1")

# Try to capture token from ?code= on redirect
capture_strava_token_from_query()

# Show Strava connection status
ath = st.session_state.get("strava_athlete") or {}
tok = st.session_state.get("strava_token")
conn_col1, conn_col2 = st.columns([1,2])
with conn_col1:
    if tok:
        st.success("Strava connected ✅")
    else:
        st.warning("Not connected to Strava")

with conn_col2:
    # Provide a login link using secrets (mirrors your existing OAuth page)
    client_id = st.secrets.get("STRAVA_CLIENT_ID")
    redirect_uri = st.secrets.get("STRAVA_REDIRECT_URI") or st.query_params.get("redirect_uri") or ""
    if client_id:
        params = {
            "client_id": client_id,
            "response_type": "code",
            "redirect_uri": redirect_uri or st.experimental_get_query_params().get("redirect_uri", [""])[0] or "",
            "approval_prompt": "auto",
            "scope": "read,read_all",
        }
        auth_url = "https://www.strava.com/oauth/authorize?" + urllib.parse.urlencode(params)
        st.link_button("Connect with Strava", auth_url, type="primary", disabled=bool(tok))
    if ath:
        st.info(f"Connected as {ath.get('firstname','')} {ath.get('lastname','')} (id {ath.get('id','?')})")

st.divider()

# -------- Data source --------
st.subheader("1) Choose your Schedule source")
mode = st.radio("Load schedule from:", ["Google Sheet (public CSV)", "Upload Excel (.xlsx)"], horizontal=True)
dfs = {}

if mode.startswith("Google"):
    default_sheet = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQlf-bVjjKJXR9qLrfcfX9It8YurrCGGypP8RroiOVsovyejlSmlBpnVu8h3Zff8Sn9_Hw6pdCiMYN9/pubhtml?gid=751090266&single=true"
    url = st.text_input("Public Google Sheet URL", value=default_sheet)
    if url:
        dfs = load_from_google_csv(url)
else:
    up = st.file_uploader("Upload master Excel (.xlsx)", type=["xlsx"])
    if up:
        dfs = load_from_uploaded_excel(up)

sched = dfs.get("Schedule")
if not isinstance(sched, pd.DataFrame) or sched.empty:
    st.info("Load your schedule to continue. The page looks for Strava route links/IDs and the Group (5k/8k) and Surface (Road/Trail).")
    st.stop()

st.success(f"Loaded Schedule with {len(sched)} rows.")

# -------- Column mapping --------
st.subheader("2) Map columns (if needed)")
# Offer best guesses, but allow overrides
cols = list(sched.columns)
def guess(colnames, keywords):
    for c in colnames:
        lc = c.lower()
        if any(k in lc for k in keywords):
            return c
    return None

group_guess = guess(cols, ["group", "distance", "5k", "8k"])
surface_guess = guess(cols, ["surface", "road", "trail", "terrain"])
url_guess = None
for c in cols:
    try:
        if (sched[c].astype(str).str.contains("strava.com", case=False, na=False)).any():
            url_guess = c; break
    except Exception:
        pass
if not url_guess:
    url_guess = guess(cols, ["url", "link", "strava"])
source_id_guess = guess(cols, ["sourceid", "source id", "route id", "id"])

c1, c2 = st.columns(2)
with c1:
    group_col = st.selectbox("Group/Distance column (5k/8k)", options=["(auto)"] + cols, index=0 if not group_guess else (cols.index(group_guess)+1))
    surface_col = st.selectbox("Surface column (Road/Trail)", options=["(auto)"] + cols, index=0 if not surface_guess else (cols.index(surface_guess)+1))
with c2:
    url_col = st.selectbox("Route URL column (Strava)", options=["(auto)"] + cols, index=0 if not url_guess else (cols.index(url_guess)+1))
    source_id_col = st.selectbox("Route ID / Source ID column (optional)", options=["(auto)"] + cols, index=0 if not source_id_guess else (cols.index(source_id_guess)+1))

group_col = None if group_col == "(auto)" else group_col
surface_col = None if surface_col == "(auto)" else surface_col
url_col = None if url_col == "(auto)" else url_col
source_id_col = None if source_id_col == "(auto)" else source_id_col

buckets, errs, dbg_df = bucket_routes(sched, group_col, surface_col, url_col, source_id_col)

with st.expander("Preview how rows were interpreted", expanded=False):
    st.dataframe(dbg_df.head(50), use_container_width=True)

if errs:
    for e in errs:
        st.warning(e)

# Show bucket summary
st.subheader("3) Buckets detected")
counts = {k: len(v) for k, v in buckets.items()}
st.write(counts)

if not st.session_state.get("strava_token"):
    st.stop()

# -------- Download & Zip --------
st.subheader("4) Download GPX and build zips")
if st.button("Fetch GPX & build ZIPs", type="primary"):
    zips, errmap = build_bucket_zips(buckets, st.session_state["strava_token"])

    if errmap:
        with st.expander("Errors (per route)", expanded=False):
            err_rows = [{"route_id": k.split(":")[-1], "bucket": k.split(":")[0] if ":" in k else "", "error": v} for k, v in errmap.items()]
            st.dataframe(pd.DataFrame(err_rows))

    # Offer downloads
    for bucket_name, blob in zips.items():
        if blob:
            st.download_button(
                label=f"Download {bucket_name}.zip",
                data=blob,
                file_name=f"{bucket_name.replace(' ', '_')}.zip",
                mime="application/zip",
            )
        else:
            st.info(f"No files for {bucket_name}")
else:
    st.info("Click the button to fetch GPX and build the ZIP files.")
