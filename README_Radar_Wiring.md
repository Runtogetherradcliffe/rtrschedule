# Radar wiring kit (roads)

Use Radar map-matching for **Road** routes, keep reverse-geocode for **Trail** routes.

## Secrets
Add to `.streamlit/secrets.toml`:
```
RADAR_API_KEY = "prj_live_xxx_your_radar_secret"
```

## Sheet
Add optional columns: **Must Roads A**, **Must Roads B** (comma-separated).

## social_posts.py sketch
```python
from radar import radar_match_steps
from roads_integration import must_include_from_sheet

season = (r.get("Season") or r.get("season") or "").strip().lower()
poly = r.get("polyline") or ""

try:
    if season == "road" and poly:
        prelist = radar_match_steps(poly, mode="foot")
    else:
        prelist = onroute_named_segments(poly)  # your existing function
except Exception:
    prelist = onroute_named_segments(poly)

must_csv = r.get("Must Roads A") if route_index == 0 else r.get("Must Roads B")
final_segments = must_include_from_sheet(prelist, prelist, must_csv or "")
sentence = describe_turns_sentence(final_segments)
```
