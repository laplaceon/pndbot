import pandas as pd
import numpy as np
from scipy import signal
import os
import pytz
from datetime import datetime, timedelta

binance_dir = "../data/full/binance"

peaks_df = {}

for i, chart_name in enumerate(os.listdir(binance_dir)):
    chart = pd.read_csv(f"{binance_dir}/{chart_name}")
    chart['datetime'] = pd.to_datetime(chart['datetime'], format="%Y-%m-%dT%H:%M:%S.%fZ")

    pump_dt = pytz.utc.localize(datetime.strptime(chart_name[chart_name.rindex("_")+1:-4], "%Y-%m-%d %H.%M"))

    left_bound = pump_dt - timedelta(minutes=30)
    right_bound = pump_dt + timedelta(minutes=10)

    subchart = chart[(chart["timestamp"] >= left_bound.timestamp() * 1000) & (chart["timestamp"] <= right_bound.timestamp() * 1000)].copy()
    pump_i = subchart.index[subchart["timestamp"] >= pump_dt.timestamp() * 1000][0]

    # subchart["price_n"] = (subchart["price"] - subchart["price"].min()) / (subchart["price"].max() - subchart["price"].min())

    accs = signal.savgol_filter(subchart["price"], window_length=17, polyorder=3, deriv=2)
    slopes = signal.savgol_filter(subchart["price"], window_length=17, polyorder=3, deriv=1)
    spikes = np.where((accs > np.percentile(accs, 50)) & (slopes > np.percentile(slopes, 50)))[0]

    # peaks = signal.find_peaks(subchart["price"].values)[0]
    peaks = signal.find_peaks_cwt(subchart["price"].values, widths=np.arange(5, 15))
    max_peak = np.argmax(subchart.iloc[peaks]["price"])

    # True pump occurred after label
    if subchart.iloc[peaks[max_peak]]["timestamp"] >= subchart.loc[pump_i]["timestamp"]:
        first_spike = spikes[(spikes < peaks[max_peak])][0]
        true_pump_start = subchart.iloc[first_spike]["datetime"]
        true_pump_end = subchart.iloc[peaks[max_peak]]["datetime"]
    else:
        true_pump_end = subchart.iloc[peaks[max_peak]]["datetime"]
        true_pump_start = subchart.iloc[spikes[0]]["datetime"]

        if true_pump_start > true_pump_end:
            true_pump_start = true_pump_end - timedelta(minutes=1)

    chart_k = chart_name[:chart_name.rindex(" ")]
    if chart_k in peaks_df:
        true_pump_old = peaks_df[chart_k]
        peaks_df[chart_k] = (min(true_pump_old[0], true_pump_start), max(true_pump_old[1], true_pump_end))
    else:
        peaks_df[chart_k] = (true_pump_start, true_pump_end)

    # peaks_df.loc[i] = [chart_name, true_pump_start, true_pump_end, 0]


landmarks = {}
for chart_name in os.listdir(binance_dir):
    chart_k = chart_name[:chart_name.rindex(" ")]

    landmarks[chart_name] = (peaks_df[chart_k][0], peaks_df[chart_k][1], 1)

landmarks = pd.DataFrame.from_dict(landmarks, orient='index', columns=['start', 'end', 'label'])
landmarks.to_csv('../data/chart_landmarks_auto_overlapped.csv')