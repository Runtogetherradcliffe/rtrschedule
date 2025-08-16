
import streamlit as st
import pandas as pd
import requests
import polyline
import time

# ---------------------------------------------------
# LocationIQ + Strava POI Fetcher
# ---------------------------------------------------

st.title("📍 Route POIs — LocationIQ")

st.markdown(
    "This tool fetches **Points of Interest (POIs)** along Strava routes "
    "using your LocationIQ API key.\n"
    "- Requires Strava OAuth connection (already in your app).\n"
    "- Add your LocationIQ key in `.streamlit/secrets.toml`.\n"
    "- Works in batches to avoid rate limits."
)

# Settings
start_row = st.number_input("Start row (0-based)", min_value=0, value=0)
rows_per_pass = st.number_input("Rows per pass", min_value=1, value=5)
sample_interval = st.number_input("Sampling interval (meters)", min_value=100, value=300)
max_samples = st.number_input("Max samples per route", min_value=5, value=30)
delay_per_call = st.number_input("Delay between LocationIQ calls (seconds)", min_value=0.0, value=0.3, step=0.1)
delay_between_routes = st.number_input("Delay between routes (seconds)", min_value=0.0, value=1.0, step=0.5)

uploaded = st.file_uploader("Upload schedule file (CSV or Excel)", type=["csv", "xlsx"])

LOCATIONIQ_API_KEY = st.secrets.get("locationiq", {}).get("api_key", None)
if not LOCATIONIQ_API_KEY:
    st.error("⚠️ Missing LocationIQ API key in Streamlit secrets!")
    st.stop()

# ---------------------------------------------------
# Helpers
# ---------------------------------------------------

def reverse_geocode(lat, lon):
    url = f"https://us1.locationiq.com/v1/reverse.php"
    params = {
        "key": LOCATIONIQ_API_KEY,
        "lat": lat,
        "lon": lon,
        "format": "json"
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            return data.get("display_name", ""), data.get("type", "")
        else:
            return None, None
    except Exception as e:
        return None, None

def decode_and_sample(poly, interval_m=300, max_points=30):
    # Decode polyline into lat/lon points
    coords = polyline.decode(poly)
    if len(coords) <= max_points:
        return coords
    # Simple sampling: evenly pick points along the list
    step = max(1, len(coords)//max_points)
    return coords[::step]

def fetch_pois_for_route(route_id):
    # Hit Strava API for route JSON (must be available in your app's session)
    from stravalib.client import Client
    token = st.session_state.get("strava_token")
    if not token:
        return "No Strava token", [], []
    client = Client(access_token=token["access_token"])
    try:
        rjson = client.protocol.get(f"/routes/{route_id}")
        poly = rjson.get("map", {}).get("summary_polyline")
        if not poly:
            return "No polyline", [], []
        coords = decode_and_sample(poly, interval_m=sample_interval, max_points=max_samples)
        names, types = [], []
        for (lat, lon) in coords:
            name, typ = reverse_geocode(lat, lon)
            if name:
                names.append(name)
            if typ:
                types.append(typ)
            time.sleep(delay_per_call)
        if not names:
            return "No POIs found", [], []
        return "OK", list(set(names)), types
    except Exception as e:
        return f"Error: {e}", [], []

# ---------------------------------------------------
# Main logic
# ---------------------------------------------------

if uploaded:
    if uploaded.name.endswith(".csv"):
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_excel(uploaded)

    st.write("Loaded file:", df.shape)

    results = []
    subset = df.iloc[start_row:start_row+rows_per_pass].copy()

    for idx, row in subset.iterrows():
        url = str(row.get("Route 1 - Route Link (Source URL)", ""))
        rid = None
        if "strava.com/routes/" in url:
            rid = url.split("/")[-1]
        if not rid:
            results.append({"Row": idx, "Route": row.get("Route 1 - Name", ""), "Status": "Skipped (non-Strava)", "POI Summary": ""})
            continue

        status, names, types = fetch_pois_for_route(rid)
        poi_summary = "; ".join(names[:10])
        top_types = pd.Series(types).value_counts().head(5).to_dict()
        results.append({
            "Row": idx,
            "Route": row.get("Route 1 - Name", ""),
            "Status": status,
            "POI Summary": poi_summary,
            "Top Categories": top_types
        })

        time.sleep(delay_between_routes)

    outdf = pd.DataFrame(results)
    st.dataframe(outdf)

    csv = outdf.to_csv(index=False).encode("utf-8")
    st.download_button("Download POIs (CSV)", data=csv, file_name="pois_results.csv", mime="text/csv")
