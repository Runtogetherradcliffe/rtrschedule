
# pages/_settings.py
# Build: v2025.08.16-SETTINGS-1
# Appears first in the sidebar due to leading underscore

import streamlit as st
from app_config import get_cfg, set_cfg, get_all, reset_all

st.set_page_config(page_title="Settings", page_icon=":gear:", layout="centered")
st.title("Settings")
st.caption("Configure defaults used across pages (session-based).")

with st.form("cfg"):
    st.subheader("Google Sheet")
    gs = st.text_input("Default Google Sheet URL", value=get_cfg("GS_URL_DEFAULT"))

    st.subheader("Meet Details")
    meet_loc = st.text_input("Default meet location", value=get_cfg("MEET_LOC_DEFAULT"))
    meet_time = st.text_input("Default meet time (24h, e.g., 19:00)", value=get_cfg("MEET_TIME_DEFAULT"))

    st.subheader("Booking Links")
    book_url = st.text_input("Run booking URL", value=get_cfg("BOOK_URL"))
    cancel_url = st.text_input("Cancellation URL", value=get_cfg("CANCEL_URL"))

    saved = st.form_submit_button("Save settings")
    if saved:
        set_cfg("GS_URL_DEFAULT", gs.strip())
        set_cfg("MEET_LOC_DEFAULT", meet_loc.strip())
        set_cfg("MEET_TIME_DEFAULT", meet_time.strip())
        set_cfg("BOOK_URL", book_url.strip())
        set_cfg("CANCEL_URL", cancel_url.strip())
        st.success("Saved for this session.")

st.divider()
st.subheader("Current values")
st.json(get_all())

st.divider()
if st.button("Reset to defaults"):
    reset_all()
    st.success("Session settings cleared. Reload pages to see defaults.")
