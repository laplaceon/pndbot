from torch.utils.data import Dataset, DataLoader
import pandas as pd
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pytz
from random import randint
import torch
import torch.nn.functional as F
from math import floor, sqrt

class ChartDataset(Dataset):
    def __init__(self, files, tf=512, multiplier=1):
        self.multiplier = multiplier
        self.tf = tf
        self.mms = MinMaxScaler()
        self.files = [None] * len(files)
        self.pump_range = [x[1] for x in files]

        for i, file in enumerate(files):
            self.files[i] = pd.read_csv(file[0])[['timestamp', 'side', 'price', 'amount']]
            self.files[i]['side'] = self.files[i]['side'].apply(lambda x: 1 if x == 'buy' else 0)

    def __getitem__(self, i):
        idx = floor(i / self.multiplier)

        x = randint(0, len(self.files[idx]) - self.tf)

        vals = self.files[idx].iloc[x:x+self.tf].copy()

        final_ts = vals.values[-1][0]

        pumping = 1 if (final_ts >= self.pump_range[idx][0] and final_ts <= self.pump_range[idx][1]) else 0

        vals[['timestamp', 'price', 'amount']] = self.mms.fit_transform(vals[['timestamp', 'price', 'amount']])

        return {
            "seq": vals.values,
            "pumping": pumping
        }

    def __len__(self):
        return len(self.files) * self.multiplier

def get_data_loaders(landmarks_file, charts_dir, bs=16, multiplier=1):
    landmarks = pd.read_csv(landmarks_file)
    pumps = landmarks[landmarks['label'] == 1]

    landmarks_map = []

    for i, row in pumps.iterrows():
        id = row['id']
        date = id[id.rindex('_')+1:]
        start = [int(x) for x in row['start'].split(":")]
        end = [int(x) for x in row['end'].split(":")]
        date = pytz.utc.localize(datetime.strptime(date.replace('.', ':'), "%Y-%m-%d %H:%M"))
        start = date.replace(hour=start[0], minute=start[1])
        # start -= timedelta(seconds=45)
        end = date.replace(hour=end[0], minute=end[1])
        # end -= timedelta(seconds=15)
        landmarks_map.append((f"{charts_dir}{id}.csv", (int(start.timestamp()) * 1000, int(end.timestamp()) * 1000)))

    X_train, X_val = train_test_split(landmarks_map, train_size=0.7, random_state=42)

    train_ds = ChartDataset(X_train, multiplier=multiplier)
    val_ds = ChartDataset(X_val, multiplier=multiplier)

    return DataLoader(train_ds, batch_size=bs, shuffle=True), DataLoader(val_ds, batch_size=bs*2)
