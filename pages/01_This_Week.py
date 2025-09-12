import streamlit as st
from sheets import load_schedule
from utils import next_run_row, format_run_row, is_no_run

st.title("This Week's Run")

with st.spinner("Loading schedule…"):
    df = load_schedule()

nr = next_run_row(df)

if nr is None:
    st.info("No upcoming run found.")
else:
    st.success("Here's the next club run:")
    st.markdown(format_run_row(nr))

    try:
        if is_no_run(nr["date"].date()):
            st.error("This date is marked as **No run** by club rules.")
    except Exception:
        pass

with st.expander("Preview data"):
    st.dataframe(df, use_container_width=True)
