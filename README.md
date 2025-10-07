
# Download GPX Zips — Fixed Columns & Shared Sheet (v2)

This version is hard-wired to your shared Google Sheet:

- **Spreadsheet ID:** `1ncT1NCbSnFsAokyFBkMWBVsk7yrJTiUfG0iBRxyUCTw`
- **Sheet tab:** `Annual_Schedule_MASTER`
- **Mappings:** Route 1 ⇒ **8k**, Route 2 ⇒ **5k**, **Mixed ⇒ Trail**
- **Include only** rows where `Route X - Route Link Type` is `Strava Route`.
- Route ID is taken from `Route X - Source ID` (preferred) or parsed from `Route X - Route Link (Source URL)`.

### Files
- `pages/download_gpx_zips.py`
- `helpers/strava_gpx.py`

### Setup
Add your Strava secrets to Streamlit:
- `STRAVA_CLIENT_ID`
- `STRAVA_CLIENT_SECRET`
- optional: `STRAVA_REDIRECT_URI`

Use your existing OAuth page or click **Connect with Strava** on this page.
