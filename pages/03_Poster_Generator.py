import io
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import streamlit as st
from sheets import load_schedule
from utils import next_run_row, is_no_run

st.title("Poster Generator")

with st.spinner("Loading schedule…"):
    df = load_schedule()

nr = next_run_row(df)
if nr is None:
    st.info("No upcoming run found.")
    st.stop()

club_name = st.text_input("Club name", value="RunTogether Radcliffe")
subtitle = st.text_input("Subtitle", value="This Week’s Run")
logo = st.file_uploader("Optional: upload logo (PNG)", type=["png"]) 

W, H = 1200, 1600
M = 80
bg = (245, 246, 250)
fg = (30, 30, 30)
accent = (0, 102, 204)

img = Image.new("RGB", (W, H), bg)
draw = ImageDraw.Draw(img)

try:
    title_font = ImageFont.truetype("arial.ttf", 96)
    h2_font = ImageFont.truetype("arial.ttf", 64)
    body_font = ImageFont.truetype("arial.ttf", 40)
except Exception:
    title_font = ImageFont.load_default()
    h2_font = ImageFont.load_default()
    body_font = ImageFont.load_default()

draw.rectangle([0, 0, W, 280], fill=accent)
draw.text((M, 90), club_name, fill=(255, 255, 255), font=title_font)

if logo is not None:
    try:
        L = Image.open(logo).convert("RGBA")
        max_h = 180
        scale = max_h / L.height
        L = L.resize((int(L.width * scale), int(L.height * scale)))
        img.paste(L, (W - M - L.width, 50), L)
    except Exception:
        pass

d = nr.get("Date") or (nr.get("date").date() if nr.get("date") is not None else "")
start = nr.get("Start", "")
route = nr.get("Route", "")
notes = nr.get("Notes", "")

lines = [subtitle, "", f"Date: {d}", "We set off at 7:00pm", f"Start: {start}", f"Route: {route}"]
if notes:
    lines += ["", f"Notes: {notes}"]

y = 360
for line in lines:
    draw.text((M, y), line, fill=fg, font=h2_font if line == subtitle else body_font)
    y += 90 if line == subtitle else 60

draw.line([(M, H-200), (W-M, H-200)], fill=accent, width=6)
draw.text((M, H-160), "Follow us on Facebook & Instagram @RunTogetherRadcliffe", fill=fg, font=body_font)

buf = io.BytesIO()
img.save(buf, format="PNG")
st.image(img, caption="Poster preview", use_column_width=True)
st.download_button("Download poster", data=buf.getvalue(), file_name="rtr_poster.png", mime="image/png")

try:
    if is_no_run(nr["date"].date()):
        st.warning("Heads up: This date is a **No run** holiday per club rules.")
except Exception:
    pass
