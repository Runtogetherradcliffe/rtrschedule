# Club Schedule Streamlit App

This Streamlit app manages and reviews your running club's annual schedule.

## Pages

### club_schedule.py
- Main page to load and review the master schedule from Google Sheets or Excel.
- Checks for route overuse, seasonal mismatches, duplication, and 'No run' rules.

### pages/club_route_links.py
- Validates links in the Route Master tab.
- Detects Strava Route / Activity, Plotaroute, and direct GPX.
- Produces a CSV validation report.
- Downloads GPX for direct `.gpx` links.

## Data Sources Supported

### Google Sheets (CSV export — recommended)
- Share your Google Sheet: 'Anyone with the link can view'.
- Required tabs: `Schedule`, `RouteMaster`, `Config`.
- Optional: `Rules`, `Pair Map`, `Fetch GPX Checklist`.

### Excel (.xlsx)
- Requires `openpyxl` (included in requirements.txt).
- Tabs: `Schedule`, `RouteMaster` (or `Route Master`), `Config`.

### CSV files
- Export each required tab as CSV and upload:
  - `Schedule.csv`
  - `RouteMaster.csv`
  - `Config.csv`
  - (optional) `Rules.csv`

## Deployment on Streamlit Cloud
1. Push all files to your GitHub repo.
2. Connect the repo to Streamlit Cloud and deploy.

