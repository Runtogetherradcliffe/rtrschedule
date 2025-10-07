
# Download GPX Zips — New Streamlit Page

This adds a new Streamlit page at `pages/download_gpx_zips.py` which:
- Reuses your existing Strava OAuth (reads/writes `st.session_state["strava_token"]` and `["strava_athlete"]`).
- Loads your Schedule (from your public Google Sheet or an uploaded Excel).
- Extracts Strava Route IDs from URL or Source ID.
- Groups rows into **Road 5k**, **Trail 5k**, **Road 8k**, **Trail 8k**.
- Deduplicates downloads across the whole schedule.
- Downloads each GPX via the Strava API and produces four ZIP files for download.

## Setup

1. Ensure Streamlit secrets contain:
   - `STRAVA_CLIENT_ID`
   - `STRAVA_CLIENT_SECRET`
   - (optional) `STRAVA_REDIRECT_URI`

2. Drop the two files into your app:
   - `pages/download_gpx_zips.py`
   - `helpers/strava_gpx.py`

3. Deploy/run. You can use either the *existing Strava OAuth page* or this page's **Connect with Strava** button.
   Both will result in a token saved in `st.session_state["strava_token"]`.

## Notes

- Only public/accessible routes can be exported. Private routes require the authorized athlete to have access.
- We avoid repeated downloads of the same route ID. If a route belongs to multiple buckets, the same GPX is reused when building zips.
- Bucket detection is heuristic; if the app doesn't auto-detect, choose the correct columns in the UI.
