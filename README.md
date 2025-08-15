
# Running Club Schedule App

This Streamlit app includes:
- `club_schedule.py`: load your schedule from Google Sheets or Excel, and run checks (overuse, season mismatches, etc.).
- `pages/club_route_links.py`: validate route links from the Schedule tab, normalize Strava/Plotaroute URLs, and export GPX (Strava via OAuth).
- `pages/strava_oauth.py`: connect your Strava account.

## Deploy (Streamlit Cloud)
1. Push all files to GitHub.
2. Set Streamlit Secrets (App → Settings → Secrets):
```
STRAVA_CLIENT_ID = "your_client_id"
STRAVA_CLIENT_SECRET = "your_client_secret"
STRAVA_REDIRECT_URI = "https://YOUR-APP.streamlit.app"
```
3. Deploy the app.

## Google Sheets mode
- Share the sheet as "Anyone with the link can view".
- Required tabs: `Schedule`; optional: `Route Master`/`RouteMaster`/`Routemaster`, `Config`.
