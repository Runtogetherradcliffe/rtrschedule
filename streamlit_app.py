import streamlit as st
from sheets import load_schedule
from utils import next_run_row, format_run_row, is_no_run

st.set_page_config(page_title="RunTogether Radcliffe — Schedule", page_icon="🏃", layout="wide")

st.title("RunTogether Radcliffe — Schedule")

with st.spinner("Loading schedule…"):
    df = load_schedule()

st.caption("Data source: hard-wired to the club Google Sheet (Schedule tab).")

if df.empty:
    st.warning("No rows found in the Schedule tab.")
    st.stop()

nr = next_run_row(df)
if nr is None:
    st.info("No upcoming runs found.")
else:
    st.subheader("Next Run")
    run_text = format_run_row(nr)
    st.write(run_text)

    try:
        if is_no_run(nr["date"].date()):
            st.error("This date is marked as **No run** by club rules (Christmas Day, Boxing Day, or New Year’s Day).")
    except Exception:
        pass

st.divider()

st.subheader("Full Schedule (from Google Sheet)")
st.dataframe(df, use_container_width=True)
