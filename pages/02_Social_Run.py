import streamlit as st
from sheets import load_schedule

st.title("Social Runs")

with st.spinner("Loading schedule…"):
    df = load_schedule()

if "Type" in df.columns:
    social = df[df["Type"].str.contains("social", case=False, na=False)].copy()
else:
    st.info("No 'Type' column found; showing full schedule instead.")
    social = df

st.dataframe(social, use_container_width=True)

st.caption("Tip: Mark runs as 'Social' in the Schedule sheet's Type column.")
