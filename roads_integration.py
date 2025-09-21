from typing import List, Dict, Any
import re

def canonical(nm: str) -> str:
    if not nm: return ""
    s = re.sub(r"^[A-Z]{1,2}\d+\s+", "", (nm or "").strip())
    s = re.sub(r"\brd\b|\brd\.\b", "Road", s, flags=re.IGNORECASE)
    s = re.sub(r"\bln\b|\bln\.\b", "Lane", s, flags=re.IGNORECASE)
    s = re.sub(r"\bave\b|\bave\.\b", "Avenue", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def must_include_from_sheet(final_segments: List[Dict[str,Any]], prelist: List[Dict[str,Any]], csv_names: str) -> List[Dict[str,Any]]:
    want = [canonical(x) for x in (csv_names or "").split(",") if x.strip()]
    if not want:
        return final_segments

    rep_by = {}
    have = set()
    for seg in prelist:
        nm = canonical(seg.get("name") or "")
        if not nm: continue
        have.add(nm.lower())
        if nm.lower() not in rep_by:
            rep_by[nm.lower()] = seg

    out = list(final_segments)
    present = {canonical(s.get("name") or "").lower() for s in out}

    for nm in want:
        k = nm.lower()
        if k in have and k not in present:
            out.append(rep_by[k])
            present.add(k)

    seen, uniq = set(), []
    for s in out:
        k = canonical(s.get("name") or "").lower()
        if k and k not in seen:
            seen.add(k); uniq.append(s)
    return uniq
