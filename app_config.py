
# app_config.py
# Minimal session-backed config helper

import streamlit as st

# Known keys and sensible defaults
DEFAULTS = {
    "GS_URL_DEFAULT": "",
    "MEET_LOC_DEFAULT": "Radcliffe Market",
    "MEET_TIME_DEFAULT": "19:00",
    "BOOK_URL": "https://groups.runtogether.co.uk/RunTogetherRadcliffe/Runs",
    "CANCEL_URL": "https://groups.runtogether.co.uk/My/BookedRuns",
}

def get_cfg(key: str, default: str | None = None):
    if default is None:
        default = DEFAULTS.get(key, "")
    return st.session_state.get(key, default)

def set_cfg(key: str, value):
    st.session_state[key] = value

def get_all():
    data = {k: get_cfg(k) for k in DEFAULTS.keys()}
    # include any extra runtime keys if needed
    for k, v in st.session_state.items():
        if k not in data and isinstance(v, (str, int, float, bool)):
            data[k] = v
    return data

def reset_all():
    for k in list(st.session_state.keys()):
        # only clear our own keys
        if k in DEFAULTS or k.endswith("_DEFAULT"):
            del st.session_state[k]
