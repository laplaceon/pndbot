import ccxt
from ccxt.base.errors import RequestTimeout
import pandas as pd
from datetime import datetime
from datetime import timedelta
import time

import numpy as np

import torch
import torch.nn.functional as F

from model import PndModel

import onnxruntime
from sklearn.preprocessing import MinMaxScaler

import torch

binance = ccxt.binance()

window_size = 1000
mms = MinMaxScaler(feature_range=(-1, 1))

# wpump = onnxruntime.InferenceSession("./models/onnx/wpump_2_rnn_tf_1.onnx")
# wpump = onnxruntime.InferenceSession("./models/onnx/wpump_2_rnn_30s_3.onnx")
# wpump = onnxruntime.InferenceSession("./models/onnx/wpump_2_cnn_30s_1.onnx")
# wpump = onnxruntime.InferenceSession("./models/onnx/wpump_2_cnn_30s_h24_2.onnx")
# wpump = onnxruntime.InferenceSession("./models/onnx/wpump_2s_h28_2.onnx")
# wpump_st = onnxruntime.InferenceSession("./models/onnx/pnd_v2s_mtl_2lh24_3.onnx")
# wpump_st = onnxruntime.InferenceSession("./models/onnx/pnd_v2s_mtl_2lh24_2.onnx")
# wpump_st = onnxruntime.InferenceSession("./models/onnx/pnd_v2s_mtl_2lh28b36.onnx")
# wpump_st = onnxruntime.InferenceSession("./models/onnx/pnd_ns_2lh24e32.onnx")
# wpump_st = onnxruntime.InferenceSession("./models/onnx/pnd_dns_2lh18e24.onnx")
# wpump = onnxruntime.InferenceSession("./models/onnx/wpump_2s_mtl_h24_2.onnx")
# wpump = onnxruntime.InferenceSession("./models/onnx/wpump_2_rnn_3.onnx")

pnd = PndModel()
pnd.load_state_dict(torch.load("../models/clstm-1.pt"))

pnd.eval()

def to_timestamp(dt):
    return binance.parse8601(dt.isoformat())

def get_all_windows(symbol, date_end):
    records = []

    while len(records) < window_size:
        records_p = []
        date_start = date_end - timedelta(hours=1)
        orders = binance.fetch_trades(symbol, to_timestamp(date_start), params={"endTime": to_timestamp(date_end)})

        for l in orders:
            records_p.append({
                'timestamp': l['timestamp'],
                'datetime': l['datetime'],
                'side': l['side'],
                'price': l['price'],
                'amount': l['amount'],
                'vol': float(l['price']) * float(l['amount']),
            })

        records = records_p + records
        date_end = date_end - timedelta(hours=1)

    df = pd.DataFrame.from_records(records)

    return df

# time_start = datetime.strptime("2021-02-14 16:55", "%Y-%m-%d %H:%M")
# time_end = datetime.strptime("2021-02-14 17:02", "%Y-%m-%d %H:%M")

time_start = datetime.strptime("2021-01-30 17:26", "%Y-%m-%d %H:%M")
time_end = datetime.strptime("2021-01-30 17:31", "%Y-%m-%d %H:%M")

# time_start = datetime.strptime("2021-01-31 17:13", "%Y-%m-%d %H:%M")
# time_end = datetime.strptime("2021-01-31 17:31", "%Y-%m-%d %H:%M")

time = time_start

while time < time_end:
    # df = get_all_windows("SKY/BTC", time)
    df = get_all_windows("NAS/ETH", time)
    # df = get_all_windows("CDT/ETH", time)

    last_date = df.iloc[-1]['datetime']

    print(last_date)

    dfi = df.iloc[-window_size:].copy()
    dt = pd.to_datetime(dfi['datetime'])
    dfi['hour'] = dt.dt.hour + (dt.dt.minute / 60.)
    dfi['hour_sin'] = np.sin(2.*np.pi*dfi['hour']/24.)
    dfi['hour_cos'] = np.cos(2.*np.pi*dfi['hour']/24.)
    dfi['sec'] = dt.dt.second + (dt.dt.microsecond * 1e-6) # Multiplication for speed
    dfi['sec_sin'] = np.sin(2.*np.pi*dfi['sec']/60.)
    dfi['sec_cos'] = np.cos(2.*np.pi*dfi['sec']/60.)
    dfi = dfi[['hour_sin', 'hour_cos', 'sec_sin', 'sec_cos', 'side', 'price', 'amount']]
    dfi['side'] = dfi['side'].apply(lambda x: 1 if x == 'buy' else -1)

    dfi[['price', 'amount']] = mms.fit_transform(dfi[['price', 'amount']])

    inputs = torch.tensor(dfi.values).float()

    with torch.no_grad():
    # pred1 = wpump_mtl.run(None, {"input": inputs})[0]
        pred2 = F.softmax(pnd(inputs.unsqueeze(0)), dim=1)[0]

        print(pred2)

        # if np.argmax(pred2.cpu().numpy(), axis=1)[0] == 0:
        #     print("mtl: Not pumping")
        # else:
        #     print("mtl: Pumping")

        if torch.argmax(pred2) == 0:
            print("st: Not pumping")
        else:
            print("st: Pumping")

        # print(dfi)

    time += timedelta(seconds=25)
