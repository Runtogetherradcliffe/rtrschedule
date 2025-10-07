
# Download GPX — Single Zip with Subfolders (v3)

- **Spreadsheet:** `1ncT1NCbSnFsAokyFBkMWBVsk7yrJTiUfG0iBRxyUCTw` — tab `Annual_Schedule_MASTER`
- **Mapping:** Route 1 ⇒ 8k, Route 2 ⇒ 5k; **Mixed ⇒ Trail**
- **Includes only:** rows with `Route X - Route Link Type` = `Strava Route`
- **Output:** ONE zip named `All_Routes_GPX.zip` with subfolders:
  - `Road 5k/`
  - `Trail 5k/`
  - `Road 8k/`
  - `Trail 8k/`
- **Deduped:** Each unique Strava route is fetched once and reused for any bucket appearances.
- Adds a small `MANIFEST.txt` with per-bucket counts.

Drop in these files:
- `pages/download_gpx_zips.py`
- `helpers/strava_gpx.py`
