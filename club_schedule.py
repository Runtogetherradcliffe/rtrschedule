
import io
import re
import urllib.parse
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Club Schedule", page_icon="🏃", layout="wide")
st.title("🏃 Club Schedule — Review & Checks")

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
    return (m, d) >= start_md or (m, d) <= end_md

def clean(x):
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
    encoded = urllib.parse.quote(sheet_name, safe="")
    export_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={encoded}"
    df = pd.read_csv(export_url)
    if len(df.columns) and df.columns[0].lower().startswith("unnamed"):
        df = df.drop(columns=[df.columns[0]])
    return df

def load_from_google_csv(url: str):
    sheet_id = extract_sheet_id(url)
    if not sheet_id:
        st.error("Could not extract Sheet ID from URL.")
        return None
    dfs = {}
    # Route Master (accept variants)
    for tab in ["Route Master", "RouteMaster", "Routemaster"]:
        try:
            df = load_google_sheet_csv(sheet_id, tab)
            if not df.empty:
                dfs["Route Master"] = df
                break
        except Exception:
            pass
    # Schedule
    try:
        df = load_google_sheet_csv(sheet_id, "Schedule")
        if not df.empty:
            dfs["Schedule"] = df
    except Exception:
        pass
    # Config
    try:
        df = load_google_sheet_csv(sheet_id, "Config")
        if not df.empty:
            dfs["Config"] = df
    except Exception:
        pass
    # Optional
    for tab in ["Rules", "Pair Map", "Fetch GPX Checklist"]:
        try:
            df = load_google_sheet_csv(sheet_id, tab)
            if not df.empty:
                dfs[tab] = df
        except Exception:
            pass
    return dfs

def load_from_excel_bytes(bts: bytes):
    xls = pd.ExcelFile(io.BytesIO(bts))
    dfs = {}
    # Route Master-like sheet
    rm_name = None
    for name in xls.sheet_names:
        if name.lower().replace(" ", "") in {"routemaster", "route_master"}:
            rm_name = name; break
    if rm_name is None:
        rm_name = "Route Master" if "Route Master" in xls.sheet_names else ("RouteMaster" if "RouteMaster" in xls.sheet_names else None)
    if rm_name:
        dfs["Route Master"] = pd.read_excel(xls, rm_name)
    if "Schedule" in xls.sheet_names:
        dfs["Schedule"] = pd.read_excel(xls, "Schedule")
    if "Config" in xls.sheet_names:
        dfs["Config"] = pd.read_excel(xls, "Config")
    for opt in ["Rules", "Pair Map", "Fetch GPX Checklist"]:
        if opt in xls.sheet_names:
            dfs[opt] = pd.read_excel(xls, opt)
    return dfs

# -----------------------------
# Data source controls
# -----------------------------
mode = st.radio(
    "Load data from:",
    ["Google Sheet (CSV export — recommended)", "Upload Excel (.xlsx)", "Upload CSV files"],
    horizontal=True
)

dfs = None
if mode == "Google Sheet (CSV export — recommended)":
    gs_url = st.text_input("Google Sheet URL")
    if gs_url:
        dfs = load_from_google_csv(gs_url)

elif mode == "Upload Excel (.xlsx)":
    up = st.file_uploader("Upload your master Excel (.xlsx)", type=["xlsx"])
    if up is not None:
        try:
            dfs = load_from_excel_bytes(up.read())
        except ImportError:
            st.error("Excel reading requires 'openpyxl'. Add to requirements.txt or use Google Sheets/CSV.")
        except Exception as e:
            st.error(f"Could not read Excel: {e}")

else:
    st.caption("Upload 'Schedule.csv', 'RouteMaster.csv' (or 'Route Master.csv'/'Routemaster.csv'), 'Config.csv' (optional: 'Rules.csv')")
    files = st.file_uploader("Upload CSVs", type=["csv"], accept_multiple_files=True)
    if files:
        name_map = {f.name.lower(): f for f in files}
        def read_csv_if(name_variants):
            for nv in name_variants:
                if nv.lower() in name_map:
                    return pd.read_csv(name_map[nv.lower()])
            return pd.DataFrame()
        dfs = {}
        dfs["Route Master"] = read_csv_if(["RouteMaster.csv", "Route Master.csv", "Routemaster.csv"])
        dfs["Schedule"] = read_csv_if(["Schedule.csv"])
        dfs["Config"] = read_csv_if(["Config.csv"])
        if "rules.csv" in name_map:
            dfs["Rules"] = pd.read_csv(name_map["rules.csv"])

if not dfs:
    st.info("Load your data to continue.")
    st.stop()

# Harmonize
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
    if "Meet Location" not in schedule.columns and "Notes" in schedule.columns:
        schedule["Meet Location"] = schedule["Notes"].apply(infer_meet_location)

# -----------------------------
# Display + checks
# -----------------------------
st.subheader("Schedule")
st.dataframe(schedule.fillna(""), use_container_width=True, hide_index=True)

st.subheader("Checks")
pairs = []
for _, r in schedule.iterrows():
    d = r.get("Date (Thu)")
    if pd.isna(d): continue
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
    if pd.isna(d): continue
    is_dark = in_dark_season(pd.to_datetime(d), dark_start, dark_end)
    for side in ["1","2"]:
        nm = clean(r.get(f"Route {side} - Name"))
        ter = clean(r.get(f"Route {side} - Terrain (Road/Trail/Mixed)"))
        if not nm or nm.lower() == "no run" or not ter: continue
        allowed = allowed_dark if is_dark else allowed_light
        if ter not in allowed:
            violations.append({"Date": pd.to_datetime(d), "Route": nm, "Terrain": ter, "Season": "Dark" if is_dark else "Light"})
season_df = pd.DataFrame(violations).sort_values("Date") if violations else pd.DataFrame(columns=["Date","Route","Terrain","Season"])

c1, c2, c3 = st.columns(3)
with c1: st.metric("Overused routes", int(overused_df.shape[0] if not overused_df.empty else 0))
with c2: st.metric("Never-used routes", int(never_used_df.shape[0] if not never_used_df.empty else 0))
with c3: st.metric("Season mismatches", int(season_df.shape[0] if not season_df.empty else 0))

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
st.caption("Accepted Route Master tab names: 'Route Master', 'RouteMaster', 'Routemaster'.")
