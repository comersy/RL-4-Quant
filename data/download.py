"""
Downloads 3 years of historical BTC options data from Deribit public API.

For each day, creates:
    data/raw/YYYY-MM-DD/
        ├── trades.csv     last trade of each option traded that day
        └── meta.json      { "spot": close BTC price for the day }

Spot is the last index_price observed during the day (close).
"""

import os
import csv
import json
import time
import requests
from datetime import datetime, timedelta, timezone


URL_TRADES = 'https://history.deribit.com/api/v2/public/get_last_trades_by_currency_and_time'
OUT_DIR    = 'data/raw'
CURRENCY   = 'BTC'
N_DAYS     = 3 * 365     # 3 years
COUNT_MAX  = 1000        # max trades per API call


def ts_ms(dt: datetime) -> int:
    return int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000)


def fetch_day(start_ms: int, end_ms: int) -> list[dict]:
    """Fetch all option trades for a day, paginating until done."""
    all_trades = []
    current_end = end_ms

    while True:
        r = requests.get(URL_TRADES, params={
            'currency':        CURRENCY,
            'kind':            'option',
            'start_timestamp': start_ms,
            'end_timestamp':   current_end,
            'count':           COUNT_MAX,
            'sorting':         'desc',
        })
        result = r.json()['result']
        trades = result['trades']
        if not trades:
            break

        all_trades.extend(trades)

        if not result['has_more']:
            break

        oldest_ts = min(t['timestamp'] for t in trades)
        current_end = oldest_ts - 1

        time.sleep(0.1)

    return all_trades


def keep_last_per_option(trades: list[dict]) -> list[dict]:
    """Keep only the most recent trade for each option (by instrument_name)."""
    last_by_instr = {}
    for t in trades:
        name = t['instrument_name']
        if name not in last_by_instr or t['timestamp'] > last_by_instr[name]['timestamp']:
            last_by_instr[name] = t
    return list(last_by_instr.values())


def save_day(date: datetime, all_trades: list[dict]):
    day_str = date.strftime('%Y-%m-%d')
    day_dir = os.path.join(OUT_DIR, day_str)
    os.makedirs(day_dir, exist_ok=True)

    if all_trades:
        # spot = last index_price of the day (before filtering)
        last_trade = max(all_trades, key=lambda t: t['timestamp'])
        spot = last_trade['index_price']

        # keep only the last trade for each option
        filtered = keep_last_per_option(all_trades)

        fieldnames = sorted({k for t in filtered for k in t.keys()})
        with open(os.path.join(day_dir, 'trades.csv'), 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(filtered)
    else:
        with open(os.path.join(day_dir, 'trades.csv'), 'w') as f:
            f.write('')
        spot = None
        filtered = []

    with open(os.path.join(day_dir, 'meta.json'), 'w') as f:
        json.dump({'spot': spot}, f)

    return len(filtered), len(all_trades)


def main():
    today      = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = today - timedelta(days=N_DAYS)

    for i in range(N_DAYS):
        day      = start_date + timedelta(days=i)
        next_day = day + timedelta(days=1)
        day_str  = day.strftime('%Y-%m-%d')

        if os.path.exists(os.path.join(OUT_DIR, day_str, 'trades.csv')):
            print(f'[{i+1}/{N_DAYS}] {day_str}  → already exists, skip')
            continue

        print(f'[{i+1}/{N_DAYS}] {day_str}  → fetching...', end=' ', flush=True)
        try:
            trades          = fetch_day(ts_ms(day), ts_ms(next_day))
            n_kept, n_total = save_day(day, trades)
            print(f'{n_kept} options ({n_total} trades total)')
        except Exception as e:
            print(f'ERROR: {e}')


if __name__ == '__main__':
    main()