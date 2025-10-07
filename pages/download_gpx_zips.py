
# pages/download_gpx_zips.py (v3 - single zip)
import streamlit as st
import pandas as pd
import urllib.parse

from helpers.strava_gpx import (
    STRAVA_SHEET_ID, SHEET_TAB,
    capture_strava_token_from_query,
    load_master_schedule,
    build_buckets_from_master,
    build_single_zip,
)

st.set_page_config(page_title="Download GPX (Single Zip)", page_icon="📦", layout="wide")
st.title("📦 Download GPX — Strava routes from Schedule (Single Zip)")
st.caption("Build: v2025.10.07-GPXZIP-3 (One zip, subfolders; Route1→8k, Route2→5k, Mixed→Trail)")

# Capture token if redirect back with ?code=
capture_strava_token_from_query()

# Strava connection status
ath = st.session_state.get("strava_athlete") or {}
tok = st.session_state.get("strava_token")

top1, top2 = st.columns([1,2])
with top1:
    if tok:
        st.success("Strava connected ✅")
    else:
        st.warning("Not connected to Strava")

with top2:
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

st.subheader("1) Data source")
st.write(f"Using shared Google Sheet **{SHEET_TAB}** from Spreadsheet ID **{STRAVA_SHEET_ID}**.")

sched = load_master_schedule()
if not isinstance(sched, pd.DataFrame) or sched.empty:
    st.stop()

st.success(f"Loaded schedule with {len(sched)} rows.")

with st.expander("Preview first 50 rows (for sanity check)"):
    st.dataframe(sched.head(50), use_container_width=True)

st.subheader("2) Build buckets (Route 1 → 8k, Route 2 → 5k, Mixed → Trail)")
buckets, errs = build_buckets_from_master(sched)
if errs:
    for e in errs:
        st.warning(e)

counts = {k: len(v) for k, v in buckets.items()}
st.write(counts)

if not tok:
    st.stop()

st.subheader("3) Download GPX and generate one ZIP (with subfolders)")
if st.button("Fetch GPX & build single ZIP", type="primary"):
    blob, errmap = build_single_zip(buckets, tok)

    if errmap:
        with st.expander("Errors (per route)", expanded=False):
            import pandas as pd
            err_rows = [{"route_id": k.split(":")[-1], "bucket": k.split(":")[0] if ":" in k else "", "error": v} for k, v in errmap.items()]
            st.dataframe(pd.DataFrame(err_rows))

    st.download_button(
        label="Download All_Routes_GPX.zip",
        data=blob,
        file_name="All_Routes_GPX.zip",
        mime="application/zip",
    )
else:
    st.info("Click the button to fetch GPX and build the ZIP file.")
