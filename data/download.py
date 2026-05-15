"""
Downloads 1 year of historical BTC options trades from Deribit public API.

For each day, creates:
    data/raw/YYYY-MM-DD/
        ├── trades.csv     all options trades of the day
        └── meta.json      { "spot": close price BTC for the day }

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
N_DAYS     = 365
COUNT_MAX  = 1000   # max trades per API call


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

        # next page: end just before the oldest trade fetched
        oldest_ts = min(t['timestamp'] for t in trades)
        current_end = oldest_ts - 1

        time.sleep(0.1)  # be nice to the API

    return all_trades


def save_day(date: datetime, trades: list[dict]):
    day_str = date.strftime('%Y-%m-%d')
    day_dir = os.path.join(OUT_DIR, day_str)
    os.makedirs(day_dir, exist_ok=True)

    # trades.csv
    if trades:
        # collect all possible fieldnames across all trades
        fieldnames = sorted({k for t in trades for k in t.keys()})
        with open(os.path.join(day_dir, 'trades.csv'), 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(trades)

        # spot = last index_price of the day (most recent timestamp)
        last_trade = max(trades, key=lambda t: t['timestamp'])
        spot = last_trade['index_price']
    else:
        # empty day (rare) - write empty file
        with open(os.path.join(day_dir, 'trades.csv'), 'w') as f:
            f.write('')
        spot = None

    # meta.json
    with open(os.path.join(day_dir, 'meta.json'), 'w') as f:
        json.dump({'spot': spot}, f)


def main():
    today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = today - timedelta(days=N_DAYS)

    for i in range(N_DAYS):
        day  = start_date + timedelta(days=i)
        next_day = day + timedelta(days=1)
        day_str = day.strftime('%Y-%m-%d')

        # skip if already downloaded
        if os.path.exists(os.path.join(OUT_DIR, day_str, 'trades.csv')):
            print(f'[{i+1}/{N_DAYS}] {day_str}  → already exists, skip')
            continue

        print(f'[{i+1}/{N_DAYS}] {day_str}  → fetching...', end=' ', flush=True)
        try:
            trades = fetch_day(ts_ms(day), ts_ms(next_day))
            save_day(day, trades)
            print(f'{len(trades)} trades')
        except Exception as e:
            print(f'ERROR: {e}')


if __name__ == '__main__':
    main()