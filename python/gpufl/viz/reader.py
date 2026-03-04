import glob
import json
import os
import pandas as pd
from typing import List

def _parse_line(line: str) -> dict:
    try: return json.loads(line)
    except: return {}

def read_events(file_pattern: str) -> List[dict]:
    files = glob.glob(file_pattern)
    all_events = []
    for fpath in files:
        if not os.path.isfile(fpath): continue
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or not line.startswith("{"): continue
                evt = _parse_line(line)
                if evt: all_events.append(evt)
    return all_events

def read_df(file_pattern: str) -> pd.DataFrame:
    events = read_events(file_pattern)
    if not events: return pd.DataFrame()

    df = pd.DataFrame(events)

    # [IMPORTANT] Fill main timestamp from start/end if missing
    if "ts_ns" not in df.columns:
        df["ts_ns"] = pd.Series([None]*len(df), dtype="float64")

    # Map ts_start_ns -> ts_ns for sorting
    if "ts_start_ns" in df.columns:
        df["ts_ns"] = df["ts_ns"].fillna(df["ts_start_ns"])
    if "start_ns" in df.columns:
        df["ts_ns"] = df["ts_ns"].fillna(df["start_ns"])

    # Coerce
    cols = ["ts_ns", "start_ns", "end_ns", "ts_start_ns", "ts_end_ns"]
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Sort
    df = df.sort_values("ts_ns").reset_index(drop=True)
    return df