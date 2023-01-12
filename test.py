import torch
import torch.nn.functional as F

import numpy as np

from torch.optim import Adam

from data import get_data_loaders
from model import PumpDiscriminator, PndModel

from tqdm import tqdm
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

train_dl, val_dl = get_data_loaders('./data/chart_landmarks_precise.csv', './data/charts/', bs=128, multiplier=256)

# model = PndModel()
# model.cuda()
# opt = Adam(model.parameters(), lr=1e-4)

for batch in tqdm(val_dl, position=0, leave=True):
    with torch.no_grad():
        inputs = batch['seq'].float().cuda()
        next = batch['next'].float().cuda()
        labels = batch['pumping'].long().cuda()

    print(labels)