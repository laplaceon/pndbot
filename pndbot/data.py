from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale
import pytz
from random import randint, randrange
import torch
import torch.nn.functional as F
from math import floor, sqrt

class ChartDataset(Dataset):
    def __init__(self, files, tf=1000, multiplier=1, bs=1):
        self.multiplier = multiplier
        self.tf = tf
        self.next_len = 125
        self.bs = bs
        self.files = [None] * len(files)
        self.names = [file[0] for file in files]
        self.bounds = [None] * len(files)

        pump_range = [(x[1][0], x[1][1]) for x in files]

        for i, file in enumerate(files):
            self.files[i] = pd.read_csv(file[0])
            dt = pd.to_datetime(self.files[i]['datetime'])
            self.files[i]['hour'] = dt.dt.hour + (dt.dt.minute / 60.)
            self.files[i]['hour_sin'] = np.sin(2.*np.pi*self.files[i]['hour']/24.)
            self.files[i]['hour_cos'] = np.cos(2.*np.pi*self.files[i]['hour']/24.)
            self.files[i]['sec'] = dt.dt.second + (dt.dt.microsecond * 1e-6) # Multiplication for speed
            self.files[i]['sec_sin'] = np.sin(2.*np.pi*self.files[i]['sec']/60.)
            self.files[i]['sec_cos'] = np.cos(2.*np.pi*self.files[i]['sec']/60.)
            self.files[i]['side'] = self.files[i]['side'].apply(lambda x: 1 if x == 'buy' else -1)

            bounds = self.files[i].index[(self.files[i]['timestamp'] >= pump_range[i][0]) & (self.files[i]['timestamp'] <= pump_range[i][1])].tolist()
            self.bounds[i] = (bounds[0], bounds[-1])

            self.files[i] = self.files[i][['hour_sin', 'hour_cos', 'sec_sin', 'sec_cos', 'side', 'price', 'amount']]

    def __getitem__(self, i):
        idx = floor(i / self.multiplier)

        if randrange(10) >= 5:
            x = randint(0, len(self.files[idx]) - self.tf - 1)
            # x = randint(0, len(self.files[idx]) - self.tf - 1 - self.next_len)
            # print(len(self.files[idx]))
        else:
            x = randint(self.bounds[idx][0], self.bounds[idx][-1])

        vals = self.files[idx].iloc[x:x+self.tf].copy()

        last_idx = x + self.tf + 1
        # next = self.files[idx].iloc[last_idx:last_idx+self.next_len].copy()
        final_idx = vals.index[-1]
        pumping = 1 if (final_idx >= self.bounds[idx][0] and final_idx < self.bounds[idx][1]) else 0

        # next[['timestamp', 'price', 'amount']] = self.mms.transform(next[['timestamp', 'price', 'amount']])

        seq = vals
        seq[['price', 'amount']] = minmax_scale(vals[['price', 'amount']])

        # print(seq)

        return {
            "seq": seq.values,
            # "feats": np.array([avg_price_change, std_price_change]),
            "pumping": pumping
        }

    def __len__(self):
        return len(self.files) * self.multiplier

def get_data_loaders(landmarks_file, charts_dir, bs=16, multiplier=1):
    landmarks = pd.read_csv(landmarks_file)
    landmarks['start'] = pd.to_datetime(landmarks['start'])
    landmarks['end'] = pd.to_datetime(landmarks['end'])
    pumps = landmarks[landmarks['label'] == 1]

    landmarks_map = []

    for i, row in pumps.iterrows():
        id = row['id']
        # date = id[id.rindex('_')+1:]
        # date = pytz.utc.localize(datetime.strptime(date.replace('.', ':'), "%Y-%m-%d %H:%M"))
        # start = pytz.utc.localize(datetime.strptime(row['start_date'] + " " + row['start_time'], "%Y-%m-%d %H:%M:%S"))
        # end = pytz.utc.localize(datetime.strptime(row['end_date'] + " " + row['end_time'], "%Y-%m-%d %H:%M:%S"))
        # start -= timedelta(seconds=45)
        # end -= timedelta(seconds=15)
        landmarks_map.append((f"{charts_dir}{id}", (int(row['start'].timestamp()) * 1000, int(row['end'].timestamp()) * 1000)))
        # print((f"{charts_dir}{id}.csv", start, (int(start.timestamp()) * 1000, int(end.timestamp()) * 1000)))

    X_train, X_val = train_test_split(landmarks_map, train_size=0.8, random_state=42)

    train_ds = ChartDataset(X_train, multiplier=multiplier, bs=bs)
    val_ds = ChartDataset(X_val, multiplier=multiplier, bs=bs)

    return DataLoader(train_ds, batch_size=bs, shuffle=True), DataLoader(val_ds, batch_size=bs, shuffle=True)
