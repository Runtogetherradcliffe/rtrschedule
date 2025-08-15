# Club Schedule Streamlit App

This Streamlit app reviews and checks your running club's annual schedule.

## Data sources supported

### 1. Google Sheet (CSV export — recommended)
- Share your Google Sheet so **"Anyone with the link can view"**
- Required tab names (case-sensitive): `Schedule`, `RouteMaster`, `Config`
- Optional: `Rules`, `Pair Map`, `Fetch GPX Checklist`
- Paste the sheet URL into the app.

### 2. Upload Excel (.xlsx)
- Requires `openpyxl` (included in requirements.txt)
- Tabs: `Schedule`, `RouteMaster` (or `Route Master`), `Config`

### 3. Upload CSV files
- Export each required tab as CSV and upload:
  - `Schedule.csv`
  - `RouteMaster.csv`
  - `Config.csv`
  - (optional) `Rules.csv`

## Deployment on Streamlit Cloud
1. Push these files to your GitHub repo:
   - `club_schedule_csv_v2.py`
   - `requirements.txt`
   - `README.md`
2. Connect the repo to Streamlit Cloud and deploy.

