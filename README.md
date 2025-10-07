
# Download GPX — Single Zip with Subfolders & Friendly Names (v4)

- **Spreadsheet:** `1ncT1NCbSnFsAokyFBkMWBVsk7yrJTiUfG0iBRxyUCTw` — tab `Annual_Schedule_MASTER`
- **Mapping:** Route 1 ⇒ 8k, Route 2 ⇒ 5k; **Mixed ⇒ Trail**
- **Friendly names:** taken from **column C** (Route 1) and **column M** (Route 2);
  also supports optional headers `Route X - Friendly Name` / `Route X - Route Name` if present.
- **Filename pattern:** `RTR - <Friendly Name>.gpx` (sanitized). If a duplicate name appears within the same bucket, the route ID is appended.
- **GPX <name> tag:** Set to the same `RTR - <Friendly Name>` so Komoot/Garmin display the friendly name on import.
- **Includes only:** rows with `Route X - Route Link Type` = `Strava Route`
- **Output:** ONE zip named `All_Routes_GPX.zip` with subfolders:
  - `Road 5k/`, `Trail 5k/`, `Road 8k/`, `Trail 8k/`
- **Deduped:** Each unique Strava route is fetched once and reused across folders if needed.
- Adds `MANIFEST.txt` with per-bucket counts.

Drop in these files:
- `pages/download_gpx_zips.py`
- `helpers/strava_gpx.py`
