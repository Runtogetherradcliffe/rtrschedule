from __future__ import annotations
import pandas as pd
from datetime import datetime, date

HOLIDAY_DATES = {(12, 25), (12, 26), (1, 1)}

def is_no_run(d: date) -> bool:
    return (d.month, d.day) in HOLIDAY_DATES

def next_run_row(df: pd.DataFrame):
    if "date" not in df.columns:
        return None
    now = datetime.now()
    future = df[df["date"] >= pd.Timestamp(now)].sort_values("date")
    if future.empty:
        return None
    return future.iloc[0]

def format_run_row(row: pd.Series) -> str:
    d = row.get("Date") or (row.get("date").date() if pd.notna(row.get("date")) else None)
    t = row.get("Time", "19:00")
    start = row.get("Start", "")
    route = row.get("Route", "")
    notes = row.get("Notes", "")

    parts = []
    if d:
        parts.append(f"**Date:** {d}")
    parts.append("**We set off at 7:00pm**")
    if start:
        parts.append(f"**Start:** {start}")
    if route:
        parts.append(f"**Route:** {route}")
    if notes:
        parts.append(f"**Notes:** {notes}")

    return "\n\n".join(parts)
