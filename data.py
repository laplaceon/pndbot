from torch.utils.data import Dataset, DataLoader
import pandas as pd
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pytz
from random import randint, randrange
import torch
import torch.nn.functional as F
from math import floor, sqrt

class ChartDataset(Dataset):
    def __init__(self, files, tf=800, multiplier=1, bs=1):
        self.multiplier = multiplier
        self.tf = tf
        self.next_len = int(tf * 0.25)
        self.mms = MinMaxScaler()
        self.bs = bs
        self.files = [None] * len(files)
        self.names = [file[0] for file in files]
        self.bounds = [None] * len(files)

        pump_range = [(x[1][0], x[1][1]) for x in files]

        for i, file in enumerate(files):
            self.files[i] = pd.read_csv(file[0])[['timestamp', 'side', 'price', 'amount']]
            self.files[i]['side'] = self.files[i]['side'].apply(lambda x: 1 if x == 'buy' else 0)

            bounds = self.files[i].index[(self.files[i]['timestamp'] >= pump_range[i][0]) & (self.files[i]['timestamp'] <= pump_range[i][1])].tolist()
            self.bounds[i] = (bounds[0], bounds[-1])

    def __getitem__(self, i):
        idx = floor(i / self.multiplier)

        if randrange(0, 10) >= 4:
            x = randint(0, len(self.files[idx]) - self.tf - 1 - self.next_len)
        else:
            x = randint(self.bounds[idx][0], self.bounds[idx][-1])

        vals = self.files[idx].iloc[x:x+self.tf].copy()
        last_idx = x + self.tf + 1
        next = self.files[idx].iloc[last_idx:last_idx+self.next_len].copy()
        final_idx = vals.index[-1]
        pumping = 1 if (final_idx >= self.bounds[idx][0] and final_idx < self.bounds[idx][1]) else 0

        vals[['timestamp', 'price', 'amount']] = self.mms.fit_transform(vals[['timestamp', 'price', 'amount']])
        next[['timestamp', 'price', 'amount']]= self.mms.transform(next[['timestamp', 'price', 'amount']])

        return {
            "seq": vals.values,
            "next": next.values,
            "pumping": pumping
        }

    def __len__(self):
        return len(self.files) * self.multiplier

def get_data_loaders(landmarks_file, charts_dir, bs=16, multiplier=1):
    landmarks = pd.read_csv(landmarks_file)
    pumps = landmarks[landmarks['label'] == 0]

    landmarks_map = []

    for i, row in pumps.iterrows():
        id = row['id']
        # date = id[id.rindex('_')+1:]
        # date = pytz.utc.localize(datetime.strptime(date.replace('.', ':'), "%Y-%m-%d %H:%M"))
        start = pytz.utc.localize(datetime.strptime(row['start_date'] + " " + row['start_time'], "%Y-%m-%d %H:%M:%S"))
        end = pytz.utc.localize(datetime.strptime(row['end_date'] + " " + row['end_time'], "%Y-%m-%d %H:%M:%S"))
        # start -= timedelta(seconds=45)
        # end -= timedelta(seconds=15)
        landmarks_map.append((f"{charts_dir}{id}.csv", (int(start.timestamp()) * 1000, int(end.timestamp()) * 1000)))
        # print((f"{charts_dir}{id}.csv", start, (int(start.timestamp()) * 1000, int(end.timestamp()) * 1000)))

    X_train, X_val = train_test_split(landmarks_map, train_size=0.7, random_state=42)

    train_ds = ChartDataset(X_train, multiplier=multiplier, bs=bs)
    val_ds = ChartDataset(X_val, multiplier=multiplier, bs=bs)

    return DataLoader(train_ds, batch_size=bs, shuffle=True), DataLoader(val_ds, batch_size=bs, shuffle=True)
