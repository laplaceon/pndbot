import torch
import torch.nn.functional as F

from data import get_data_loaders
from model import PumpDiscriminator

train_dl, val_dl = get_data_loaders('./data/chart_landmarks.csv', './data/charts/', bs=256)
model = PumpDiscriminator()

for i in range(200):
    model.eval()
    for batch in val_dl:
        with torch.no_grad():
            print(batch)

    model.train()
    for batch in train_dl:
        print(batch)
