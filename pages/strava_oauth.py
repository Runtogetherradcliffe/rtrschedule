
import urllib.parse, requests, streamlit as st

st.set_page_config(page_title="Strava OAuth", page_icon="🔑", layout="centered")
st.title("🔑 Connect Strava")

st.markdown("""
Connect your Strava account so the app can access public route GPX via the Strava API.

**Setup (one-time):**
1. Create an app at https://www.strava.com/settings/api
2. Set the **Authorization Callback Domain** to your Streamlit app's domain (e.g. `your-app-name.streamlit.app`)
3. In Streamlit Cloud → Settings → Secrets add:
   - `STRAVA_CLIENT_ID`
   - `STRAVA_CLIENT_SECRET`
   - (optional) `STRAVA_REDIRECT_URI` to pre-fill below
""")

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

ath = st.session_state.get("strava_athlete") or {}
if ath:
    st.success("Connected as {} {} (id {})".format(ath.get("firstname",""), ath.get("lastname",""), ath.get("id","?")))
else:
    st.info("After authorizing on Strava, you'll be redirected back and the app will capture your token automatically.")
