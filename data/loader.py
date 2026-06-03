"""
Loads downloaded Deribit data from data/raw/ into usable structures.

For each day, returns:
    spot    : float, BTC close price of the day
    options : list of dicts, one per unique option traded that day
        {
            "instrument":  "BTC-4JAN25-92000-P",
            "strike":      92000.0,
            "expiry":      datetime,
            "option_type": "call" | "put",
            "price":       float,   # in BTC, last trade of the day
            "iv":          float,   # implied vol %
            "volume":      float,   # contracts traded
        }
"""

import csv
import json
import re
from datetime import datetime
from pathlib import Path


RAW_DIR = Path(__file__).resolve().parent / "raw"

MONTHS = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4,  "MAY": 5,  "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}


def parse_instrument(name: str) -> dict | None:
    """Parse a Deribit option name like 'BTC-4JAN25-92000-P'."""
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
    if not RAW_DIR.exists():
        return []
    return sorted(d.name for d in RAW_DIR.iterdir() if d.is_dir())


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
        "options": list of option dicts
    }
    """
    day_dir = RAW_DIR / date_str
    meta_path = day_dir / "meta.json"
    csv_path = day_dir / "trades.csv"

    with open(meta_path) as f:
        meta = json.load(f)

    options = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = parse_instrument(row["instrument_name"])
            if parsed is None:
                continue
            options.append({
                "instrument":  row["instrument_name"],
                "strike":      parsed["strike"],
                "expiry":      parsed["expiry"],
                "option_type": parsed["option_type"],
                "price":       float(row["price"]),
                "iv":          float(row["iv"]) if row["iv"] else 0.0,
                "volume":      float(row["amount"]),
            })

    return {
        "date":    date_str,
        "spot":    meta["spot"],
        "options": options,
    }



def max_options_per_day() -> int:
    """Scan all available days and return the max number of options seen in one day."""
    days = list_available_days()
    return max((len(load_day(d)["options"]) for d in days), default=0)



if __name__ == "__main__":
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
                  f"K={o['strike']:.0f}  price={o['price']:.4f}  "
                  f"iv={o['iv']:.1f}%  vol={o['volume']:.1f}")
