from functools import lru_cache
import pandas as pd

SHEET_ID = "1ncT1NCbSnFsAokyFBkMWBVsk7yrJTiUfG0iBRxyUCTw"
SCHEDULE_GID = 751090266  # "Schedule" tab

CSV_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={SCHEDULE_GID}"

@lru_cache(maxsize=1)
def load_schedule() -> pd.DataFrame:
    """Load the Schedule tab as a DataFrame (hard-wired, no user input)."""
    df = pd.read_csv(CSV_URL)

    rename_map = {
        "date": "Date", "run_date": "Date",
        "time": "Time", "start_time": "Time",
        "start": "Start", "start_location": "Start",
        "route": "Route",
        "notes": "Notes",
        "type": "Type",
    }
    lower_cols = {c.lower(): c for c in df.columns}
    for src_lower, dst in rename_map.items():
        if src_lower in lower_cols:
            df.rename(columns={lower_cols[src_lower]: dst}, inplace=True)

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date
    if "Time" in df.columns:
        try:
            df["Time"] = pd.to_datetime(df["Time"], errors="coerce").dt.strftime("%H:%M")
        except Exception:
            pass

    if "Date" in df.columns:
        if "Time" in df.columns:
            dt = pd.to_datetime(df["Date"].astype(str) + " " + df["Time"].astype(str), errors="coerce")
        else:
            dt = pd.to_datetime(df["Date"].astype(str), errors="coerce")
        df.insert(0, "date", dt)

    return df
