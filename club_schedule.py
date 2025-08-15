
import io, re, urllib.parse, pandas as pd, streamlit as st

import streamlit as st, requests

def capture_strava_token_from_query():
    """If the URL contains ?code=... from Strava and we don't yet have a token,
    exchange it here so any page can complete OAuth."""
    if st.session_state.get("strava_token"):
        return
    code = st.query_params.get("code")
    if not code:
        return
    client_id = st.secrets.get("STRAVA_CLIENT_ID")
    client_secret = st.secrets.get("STRAVA_CLIENT_SECRET")
    if not client_id or not client_secret:
        st.warning("Strava 'code' detected but API secrets are missing in Streamlit Secrets.")
        return
    token_url = "https://www.strava.com/oauth/token"
    payload = {
        "client_id": client_id,
        "client_secret": client_secret,
        "code": code,
        "grant_type": "authorization_code",
    }
    try:
        resp = requests.post(token_url, data=payload, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        st.session_state["strava_token"] = data.get("access_token")
        st.session_state["strava_athlete"] = data.get("athlete", {})
        # Remove ?code from URL so refreshes don't re-exchange
        qp = dict(st.query_params)
        if "code" in qp:
            del qp["code"]
            st.query_params.clear()
            for k, v in qp.items():
                st.query_params[k] = v
        st.success("Strava connected")
    except Exception as e:
        st.error("Strava token exchange failed: {}".format(e))

st.set_page_config(page_title="Club Schedule", page_icon="🏃", layout="wide")
st.title("🏃 Club Schedule — Review & Checks")

# Capture token if redirected here after OAuth
capture_strava_token_from_query()

def extract_sheet_id(url):
    m = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", url)
    return m.group(1) if m else None

def load_google_sheet_csv(sheet_id, sheet_name):
    u = "https://docs.google.com/spreadsheets/d/{}/gviz/tq?tqx=out:csv&sheet=".format(sheet_id) + urllib.parse.quote(sheet_name, safe="")
    df = pd.read_csv(u)
    if len(df.columns) and str(df.columns[0]).lower().startswith("unnamed"):
        df = df.drop(columns=[df.columns[0]])
    return df

mode = st.radio("Load data from:", ["Google Sheet (CSV export — recommended)", "Upload Excel (.xlsx)"], horizontal=True)

dfs = {}
if mode.startswith("Google"):
    url = st.text_input("Google Sheet URL")
    if url:
        sid = extract_sheet_id(url)
        if sid:
            try:
                dfs["Schedule"] = load_google_sheet_csv(sid, "Schedule")
                for tab in ["Route Master","RouteMaster","Routemaster","Config"]:
                    try:
                        dfs[tab] = load_google_sheet_csv(sid, tab)
                    except Exception:
                        pass
            except Exception as e:
                st.error("Could not load Google Sheet: {}".format(e))
else:
    up = st.file_uploader("Upload master Excel (.xlsx)", type=["xlsx"])
    if up:
        xls = pd.ExcelFile(io.BytesIO(up.read()))
        for tab in xls.sheet_names:
            dfs[tab] = pd.read_excel(xls, tab)

if "Schedule" not in dfs or dfs["Schedule"].empty:
    st.info("Load your data to continue.")
    st.stop()

schedule = dfs["Schedule"]
st.dataframe(schedule.fillna(""), use_container_width=True, hide_index=True)

ath = st.session_state.get("strava_athlete") or {}
if ath:
    st.caption("Connected to Strava as: {} {} (id {})".format(ath.get("firstname",""), ath.get("lastname",""), ath.get("id","?")))
