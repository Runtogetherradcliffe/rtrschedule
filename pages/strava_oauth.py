
import urllib.parse
import requests
import streamlit as st

st.set_page_config(page_title="Strava OAuth", page_icon="🔑", layout="centered")
st.title("🔑 Connect Strava")

st.markdown(\"\"\"
To enable **Strava route validation & GPX export**, connect your Strava account.

**Setup (one-time, owner/admin):**
1. Go to [Strava API settings](https://www.strava.com/settings/api) and create an app.
2. Set **Callback/Redirect URI** to your Streamlit app's base URL (e.g. `https://your-app.streamlit.app`).
3. Put your credentials into Streamlit **Secrets**:
   - `STRAVA_CLIENT_ID`
   - `STRAVA_CLIENT_SECRET`
   - (optional) `STRAVA_REDIRECT_URI` (defaults to the value you enter below)

On Streamlit Cloud, go to **App → Settings → Secrets** and add:
```
STRAVA_CLIENT_ID = "12345"
STRAVA_CLIENT_SECRET = "xxxxxxxxxxxxxxxx"
STRAVA_REDIRECT_URI = "https://your-app.streamlit.app"
```
Then return here and click **Connect with Strava**.
\"\"\")

client_id = st.secrets.get("STRAVA_CLIENT_ID")
client_secret = st.secrets.get("STRAVA_CLIENT_SECRET")

if not client_id or not client_secret:
    st.error("Missing `STRAVA_CLIENT_ID` or `STRAVA_CLIENT_SECRET` in Streamlit secrets.")
    st.stop()

redirect_default = st.secrets.get("STRAVA_REDIRECT_URI", "https://your-app.streamlit.app")
redirect_uri = st.text_input("Redirect URI (must match your Strava app settings)", value=redirect_default)

params = {
    "client_id": client_id,
    "response_type": "code",
    "redirect_uri": redirect_uri,
    "approval_prompt": "auto",
    "scope": "read,read_all",
}
auth_url = "https://www.strava.com/oauth/authorize?" + urllib.parse.urlencode(params)

st.link_button("Connect with Strava", auth_url, type="primary")

# Handle callback code
query_params = st.experimental_get_query_params()
code = query_params.get("code", [None])[0]
error = query_params.get("error", [None])[0]

if error:
    st.error(f"Strava returned error: {error}")

if code and (not st.session_state.get("strava_token")):
    st.info("Exchanging authorization code for access token...")
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
        st.success("Strava connected! Go back to the Route Links page to export GPX for Strava routes.")
        st.json({"athlete": st.session_state.get("strava_athlete", {})})
    except Exception as e:
        st.error(f"Token exchange failed: {e}")
