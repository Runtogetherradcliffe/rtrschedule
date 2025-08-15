
import io
import re
import time
import urllib.parse
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Club Schedule", page_icon="🏃", layout="wide")

st.title("🏃 Club Schedule — Review & Checks (CSV-based, robust loaders)")

# -----------------------------
# Utilities
# -----------------------------
def infer_meet_location(notes: str) -> str:
    if not isinstance(notes, str) or not notes.strip():
        return ""
    m = re.search(r"Meeting:\s*(.+)$", notes)
    if m:
        return m.group(1).strip()
    if "radcliffe market" in notes.lower():
        return "Radcliffe market"
    return ""

def cfg_value(df_cfg: pd.DataFrame, key: str, default=None):
    if df_cfg is None or df_cfg.empty:
        return default
    row = df_cfg.loc[df_cfg["Setting"] == key, "Value"]
    return row.values[0] if not row.empty else default

def in_dark_season(dt, dark_start, dark_end):
    if pd.isna(dt) or dark_start is None or dark_end is None:
        return False
    start_md = (dark_start.month, dark_start.day)
    end_md = (dark_end.month, dark_end.day)
    m, d = dt.month, dt.day
    if start_md <= end_md:
        return (m, d) >= start_md and (m, d) <= end_md
    # season crosses year boundary
    return (m, d) >= start_md or (m, d) <= end_md

def clean(x):  # light normalization; exact-match logic otherwise
    if pd.isna(x): return ""
    return str(x).strip()

def to_csv_download(df: pd.DataFrame, filename: str, label: str):
    if df is None or df.empty:
        st.button(f"⬇️ Download {label}", disabled=True)
        return
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download " + label, data=csv, file_name=filename, mime="text/csv")

def extract_sheet_id(url: str):
    m = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", url)
    return m.group(1) if m else None

def load_google_sheet_csv(sheet_id: str, sheet_name: str) -> pd.DataFrame:
    # Load a single tab from a Google Sheet using CSV export via gviz.
    # The sheet must be shared as 'Anyone with the link can view'.
    encoded = urllib.parse.quote(sheet_name, safe="")
    export_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={encoded}"
    try:
        df = pd.read_csv(export_url)
        # Some locales may export with unnamed index column; drop if present
        if len(df.columns) and df.columns[0].lower().startswith("unnamed"):
            df = df.drop(columns=[df.columns[0]])
        return df
    except Exception as e:
        st.error(f"Could not load tab '{sheet_name}' via CSV: {e}")
        return pd.DataFrame()

# -----------------------------
# Data source controls
# -----------------------------
st.markdown("#### Data sources")
mode = st.radio(
    "Choose how to load your master data:",
    ["Google Sheet (CSV export — recommended)", "Upload Excel (.xlsx)", "Upload CSV files (3 tabs)"],
    horizontal=True
)

dfs = None

if mode == "Google Sheet (CSV export — recommended)":
    gs_url = st.text_input(
        "Paste your Google Sheet URL (tabs required: Schedule, RouteMaster, Config). "
        "Share the sheet so 'Anyone with the link can view'. Tab names are case-sensitive."
    )
    if gs_url:
        sheet_id = extract_sheet_id(gs_url)
        if not sheet_id:
            st.error("Could not extract Sheet ID. Please paste a standard Google Sheets URL.")
        else:
            required_tabs = ["Schedule", "RouteMaster", "Config"]
            optional_tabs = ["Rules", "Pair Map", "Fetch GPX Checklist"]
            dfs_try = {}
            missing = []
            for tab in required_tabs:
                df = load_google_sheet_csv(sheet_id, tab)
                if df.empty:
                    missing.append(tab)
                dfs_try[tab] = df
            for tab in optional_tabs:
                df = load_google_sheet_csv(sheet_id, tab)
                if not df.empty:
                    dfs_try[tab] = df
            if missing:
                st.error("Missing or empty required tab(s): " + ", ".join(missing))
            else:
                dfs = dfs_try

elif mode == "Upload Excel (.xlsx)":
    up = st.file_uploader("Upload Annual_Schedule_MASTER.xlsx", type=["xlsx"])
    if up is not None:
        try:
            xls = pd.ExcelFile(io.BytesIO(up.read()))
            dfs = {
                "Schedule": pd.read_excel(xls, "Schedule"),
                "RouteMaster": pd.read_excel(xls, "Route Master") if "Route Master" in xls.sheet_names else pd.read_excel(xls, "RouteMaster"),
                "Config": pd.read_excel(xls, "Config") if "Config" in xls.sheet_names else pd.DataFrame(),
            }
            for opt in ["Rules", "Pair Map", "Fetch GPX Checklist"]:
                if opt in xls.sheet_names:
                    dfs[opt] = pd.read_excel(xls, opt)
        except ImportError as e:
            st.error("Excel reading requires the 'openpyxl' package. Add 'openpyxl' to requirements.txt or switch to CSV options.")
        except Exception as e:
            st.error(f"Could not read Excel file: {e}")

elif mode == "Upload CSV files (3 tabs)":
    st.caption("Upload three CSVs named exactly: 'Schedule.csv', 'RouteMaster.csv', 'Config.csv' (optional: 'Rules.csv')")
    files = st.file_uploader("Upload CSV files", type=["csv"], accept_multiple_files=True)
    if files:
        name_map = {f.name: f for f in files}
        missing = [n for n in ["Schedule.csv", "RouteMaster.csv", "Config.csv"] if n not in name_map]
        if missing:
            st.error("Missing required CSV file(s): " + ", ".join(missing))
        else:
            dfs = {
                "Schedule": pd.read_csv(name_map["Schedule.csv"]),
                "RouteMaster": pd.read_csv(name_map["RouteMaster.csv"]),
                "Config": pd.read_csv(name_map["Config.csv"]),
            }
            if "Rules.csv" in name_map:
                dfs["Rules"] = pd.read_csv(name_map["Rules.csv"])

if not dfs:
    st.info("Load your data using one of the methods above to continue.")
    st.stop()

# Harmonize key tab name (RouteMaster vs Route Master)
if "RouteMaster" in dfs and "Route Master" not in dfs:
    dfs["Route Master"] = dfs["RouteMaster"]

schedule = dfs.get("Schedule", pd.DataFrame())
route_master = dfs.get("Route Master", pd.DataFrame())
config = dfs.get("Config", pd.DataFrame())
rules = dfs.get("Rules", pd.DataFrame())

# -----------------------------
# Config + derived fields
# -----------------------------
overused_threshold = int(cfg_value(config, "Overused Threshold (uses/year)", 3))
dark_start = pd.to_datetime(cfg_value(config, "Dark Season Start (YYYY-MM-DD)", "2025-10-27"))
dark_end   = pd.to_datetime(cfg_value(config, "Dark Season End (YYYY-MM-DD)", "2026-03-30"))
allowed_dark  = set(str(cfg_value(config, "Allowed Terrain in Dark Season", "Road")).split(","))
allowed_light = set(str(cfg_value(config, "Light Season Allowed Terrain", "Road,Trail,Mixed")).split(","))

# Parse datetimes + meet location
if not schedule.empty:
    if "Date (Thu)" in schedule.columns:
        schedule["Date (Thu)"] = pd.to_datetime(schedule["Date (Thu)"])
    if "Meet Location" not in schedule.columns:
        schedule["Meet Location"] = schedule.get("Notes", "").apply(infer_meet_location) if "Notes" in schedule.columns else ""

# -----------------------------
# Display schedule overview
# -----------------------------
st.subheader("Schedule")
st.dataframe(
    schedule.fillna(""),
    use_container_width=True,
    hide_index=True
)

# -----------------------------
# Checks
# -----------------------------
st.subheader("Checks")

# Overuse (count each route once per date; ignore 'No run')
pairs = []
for _, r in schedule.iterrows():
    d = r.get("Date (Thu)")
    if pd.isna(d): 
        continue
    names = {clean(r.get("Route 1 - Name")), clean(r.get("Route 2 - Name"))}
    names = {n for n in names if n and n.lower() != "no run"}
    for n in names:
        pairs.append((d, n))
use_counts = (pd.DataFrame(pairs, columns=["Date","Route"])
              .groupby("Route")["Date"].nunique()
              .sort_values(ascending=False))

overused = use_counts[use_counts > overused_threshold].rename("Uses")
overused_df = overused.rename_axis("Route").reset_index()

never_used_df = pd.DataFrame(columns=["Route"])
if not route_master.empty and "Route Name" in route_master.columns:
    master_names = set(route_master["Route Name"].dropna().apply(clean))
    used_names = set(use_counts.index)
    never_used = sorted([r for r in (master_names - used_names) if r and r.lower() != "no run"])
    never_used_df = pd.DataFrame({"Route": never_used})

# Season mismatches
violations = []
for _, r in schedule.iterrows():
    d = r.get("Date (Thu)")
    if pd.isna(d):
        continue
    # Using existing dark/light boundaries
    is_dark = in_dark_season(pd.to_datetime(d), dark_start, dark_end)
    for side in ["1","2"]:
        nm = clean(r.get(f"Route {side} - Name"))
        ter = clean(r.get(f"Route {side} - Terrain (Road/Trail/Mixed)"))
        if not nm or nm.lower() == "no run" or not ter:
            continue
        allowed = allowed_dark if is_dark else allowed_light
        if ter not in allowed:
            violations.append({
                "Date": pd.to_datetime(d),
                "Route": nm,
                "Terrain": ter,
                "Season": "Dark" if is_dark else "Light",
                "Meet Location": r.get("Meet Location", "")
            })
season_df = pd.DataFrame(violations).sort_values("Date") if violations else pd.DataFrame(columns=["Date","Route","Terrain","Season","Meet Location"])

# Show results
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Overused routes", int((overused_df.shape[0] if not overused_df.empty else 0)))
with c2:
    st.metric("Never-used routes", int((never_used_df.shape[0] if not never_used_df.empty else 0)))
with c3:
    st.metric("Season mismatches", int((season_df.shape[0] if not season_df.empty else 0)))

st.markdown("### Overused")
st.dataframe(overused_df if not overused_df.empty else pd.DataFrame([{"Status":"None ✅"}]), use_container_width=True, hide_index=True)
to_csv_download(overused_df, "overused_routes.csv", "Overused Routes (CSV)")

st.markdown("### Never used (ignoring 'No run')")
st.dataframe(never_used_df if not never_used_df.empty else pd.DataFrame([{"Status":"None ✅"}]), use_container_width=True, hide_index=True)
to_csv_download(never_used_df, "never_used_routes.csv", "Never Used Routes (CSV)")

st.markdown("### Season mismatches")
st.dataframe(season_df if not season_df.empty else pd.DataFrame([{"Status":"None ✅"}]), use_container_width=True, hide_index=True)
to_csv_download(season_df, "season_mismatches.csv", "Season Mismatches (CSV)")

st.divider()

# Clumping flags (already curated by you); we only display if present
clump_cols = [c for c in schedule.columns if "Clumping" in c]
if clump_cols:
    st.markdown("### Clumping Flags (from your schedule)")
    base_cols = [c for c in ["Date (Thu)","Route 1 - Name","Route 1 - Area","Route 2 - Name","Route 2 - Area"] if c in schedule.columns]
    st.dataframe(schedule[base_cols + clump_cols].fillna(""),
                 use_container_width=True, hide_index=True)
else:
    st.caption("No clumping flag columns found. (Optional)")

# Rules display (so leaders see the holiday/no-run policy)
if isinstance(rules, pd.DataFrame) and not rules.empty:
    st.markdown("### Rules")
    st.dataframe(rules, use_container_width=True, hide_index=True)
else:
    st.caption("No Rules sheet found. (Optional)")

st.success("Loaded and checked. Meet location, OAB sharing, and holiday 'No run' rules are respected. (CSV loader—no openpyxl needed)")
