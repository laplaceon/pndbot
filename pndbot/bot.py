import ccxt
from ccxt.base.errors import RequestTimeout
import pandas as pd
from datetime import datetime
from datetime import timedelta
import time

import numpy as np

import threading

import torch
import torch.nn.functional as F

from model import PndModel

from sklearn.preprocessing import minmax_scale

import concurrent.futures

MAX_THREADS = 8

market_lock = threading.Lock()
ccxt_lock = threading.Lock()

binance = ccxt.binance()

window_size = 1000

pnd = PndModel()
pnd.load_state_dict(torch.load("../models/clstm-f61b256.pt"))

pnd.eval()

def to_timestamp(dt):
    return binance.parse8601(dt.isoformat())

def get_all_windows(symbol):
    records = []

    # ccxt_lock.acquire()
    orders = binance.fetch_trades(symbol, limit=window_size)
    # ccxt_lock.release()

    for l in orders:
        records.append({
            'datetime': l['datetime'],
            'side': l['side'],
            'price': l['price'],
            'amount': l['amount'],
        })

    df = pd.DataFrame.from_records(records)

    return df

class Bot():
    def get_features_from_df(df):
        dfi = df.copy()
        dt = pd.to_datetime(dfi['datetime'])
        dfi['hour'] = dt.dt.hour + (dt.dt.minute / 60.)
        dfi['hour_sin'] = np.sin(2.*np.pi*dfi['hour']/24.)
        dfi['hour_cos'] = np.cos(2.*np.pi*dfi['hour']/24.)
        dfi['sec'] = dt.dt.second + (dt.dt.microsecond * 1e-6) # Multiplication for speed
        dfi['sec_sin'] = np.sin(2.*np.pi*dfi['sec']/60.)
        dfi['sec_cos'] = np.cos(2.*np.pi*dfi['sec']/60.)
        dfi = dfi[['hour_sin', 'hour_cos', 'sec_sin', 'sec_cos', 'side', 'price', 'amount']]
        dfi['side'] = dfi['side'].apply(lambda x: 1 if x == 'buy' else -1)
        dfi[['price', 'amount']] = minmax_scale(dfi[['price', 'amount']])

        return dfi.values

    def __init__(self, clf) -> None:
        self.clf = clf
        self.markets = [market['symbol'] for market in binance.fetch_markets() if market['active'] and market['type'] == 'spot' and market['quote'] in ['BTC', 'ETH']]
    
    def get_markets(self) -> list:
        return self.markets

bot = Bot(pnd)

market_lock = threading.Lock()

def download_trades(market):
    df = get_all_windows(market)

    last_date = df.iloc[-1]['datetime']

    features = Bot.get_features_from_df(df)

    market_lock.acquire()

    tradesMap = {
        "market": market,
        "last_date": last_date,
        "features": features
    }

    trades.append(tradesMap)
    market_lock.release()

while True:
    trades = []

    markets = bot.get_markets()
    threads = min(MAX_THREADS, len(markets))

    t0 = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        executor.map(download_trades, markets)

    features = np.array([x["features"] for x in trades])
    inputs = torch.tensor(features).float()

    with torch.no_grad():
        preds = F.softmax(pnd(inputs), dim=1)
        labels = torch.argmax(preds, dim=1)

        probs = preds[:, 1]

        for i, label in enumerate(labels):
            market = trades[i]["market"]
            last_date = trades[i]["last_date"]

            if label == 0:
                state = "not pumping"
            else:
                state = "pumping"
                
                print(f"{market} is {state} with p(pumping) = {probs[i].item():.4f} as of {last_date}")

    t1 = time.time()
    print(f"{t1-t0} seconds to download.")

    # time.sleep(5)
