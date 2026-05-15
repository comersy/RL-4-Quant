"""
Loads downloaded Deribit data from data/raw/ into usable structures.

For each day, returns:
    spot    : float, BTC close price of the day
    options : list of dicts, one per unique option (strike + maturity + type)
        {
            "instrument": "BTC-4JAN25-92000-P",
            "strike":      92000.0,
            "expiry":      datetime,
            "option_type": "call" | "put",
            "avg_price":   float,   # in BTC
            "last_price":  float,   # in BTC
            "iv":          float,   # implied vol %
            "volume":      float,   # total contracts traded
        }
"""

import os
import csv
import json
import re
from datetime import datetime
from collections import defaultdict


RAW_DIR = "data/raw"

# month abbreviations used by Deribit instrument names
MONTHS = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4,  "MAY": 5,  "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}


def parse_instrument(name: str) -> dict | None:
    """
    Parse a Deribit option name like 'BTC-4JAN25-92000-P' into its components.
    Returns None if the format does not match (e.g. perpetual or future).
    """
    m = re.match(r"^([A-Z]+)-(\d{1,2})([A-Z]{3})(\d{2})-(\d+)-([CP])$", name)
    if not m:
        return None
    _, day, mon, year, strike, kind = m.groups()
    expiry = datetime(2000 + int(year), MONTHS[mon], int(day))
    return {
        "strike":      float(strike),
        "expiry":      expiry,
        "option_type": "call" if kind == "C" else "put",
    }


def list_available_days() -> list[str]:
    """Return all dates available in data/raw/, sorted ascending."""
    if not os.path.exists(RAW_DIR):
        return []
    return sorted(d for d in os.listdir(RAW_DIR)
                  if os.path.isdir(os.path.join(RAW_DIR, d)))


def load_day(date_str: str) -> dict:
    """
    Load one day of data.

    Parameters
    ----------
    date_str : "YYYY-MM-DD"

    Returns
    -------
    {
        "date":    str,
        "spot":    float,
        "options": list of option dicts (aggregated by instrument)
    }
    """
    day_dir   = os.path.join(RAW_DIR, date_str)
    meta_path = os.path.join(day_dir, "meta.json")
    csv_path  = os.path.join(day_dir, "trades.csv")

    with open(meta_path) as f:
        meta = json.load(f)

    # accumulate trades per instrument
    by_instr = defaultdict(list)
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            by_instr[row["instrument_name"]].append(row)

    options = []
    for name, trades in by_instr.items():
        parsed = parse_instrument(name)
        if parsed is None:
            continue  # skip non-option instruments

        prices = [float(t["price"]) for t in trades]
        ivs    = [float(t["iv"])    for t in trades if t["iv"]]
        amounts = [float(t["amount"]) for t in trades]
        # last trade = most recent timestamp
        last = max(trades, key=lambda t: int(t["timestamp"]))

        options.append({
            "instrument":  name,
            "strike":      parsed["strike"],
            "expiry":      parsed["expiry"],
            "option_type": parsed["option_type"],
            "avg_price":   sum(prices) / len(prices),
            "last_price":  float(last["price"]),
            "iv":          sum(ivs) / len(ivs) if ivs else 0.0,
            "volume":      sum(amounts),
        })

    return {
        "date":    date_str,
        "spot":    meta["spot"],
        "options": options,
    }


if __name__ == "__main__":
    # quick test
    days = list_available_days()
    print(f"Found {len(days)} days in {RAW_DIR}")
    if days:
        d = load_day(days[0])
        print(f"\nDate:     {d['date']}")
        print(f"Spot:     ${d['spot']:.2f}")
        print(f"Options:  {len(d['options'])} unique instruments")
        print("\nFirst 3 options:")
        for o in d["options"][:3]:
            print(f"  {o['instrument']:30}  {o['option_type']:4}  "
                  f"K={o['strike']:.0f}  avg={o['avg_price']:.4f}  "
                  f"last={o['last_price']:.4f}  iv={o['iv']:.1f}%  vol={o['volume']:.1f}")