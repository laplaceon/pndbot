import torch
import torch.nn.functional as F

import numpy as np

from torch.optim import Adam

from data import get_data_loaders
from model import PumpDiscriminator

from tqdm import tqdm
from sklearn.metrics import f1_score, matthews_corrcoef, accuracy_score

train_dl, val_dl = get_data_loaders('./data/chart_landmarks.csv', './data/charts/', bs=256, multiplier=128)
model = PumpDiscriminator()
opt = Adam(model.parameters(), lr=1e-3)

def compute_accuracy(pred, labels):
    pred = np.argmax(pred, axis=1)

    return {
        "acc": accuracy_score(labels, pred),
        "f1": f1_score(labels, pred),
        "mcc": matthews_corrcoef(labels, pred)
    }

for i in range(200):
    model.eval()
    valid_stats = 2 * [[0, 0, 0, 0]]
    for batch in tqdm(val_dl, position=0, leave=True):
        with torch.no_grad():
            inputs = batch['seq'].float()
            labels = batch['pumping'].long()

            preds = model(inputs)

            loss_0 = F.cross_entropy(preds, labels)

            scores = compute_accuracy(preds.detach().cpu().numpy(), labels.detach().cpu().numpy())

            valid_stats[0] = [a + b for a, b in zip(valid_stats[0], [loss_0.item(), scores["acc"], scores["f1"], scores["mcc"]])]
            valid_stats[1] = [a + b for a, b in zip(valid_stats[1], [0, 0, 0, 0])]
    nb = len(val_dl)
    for i in valid_stats:
        print(f'Avg. loss: {i[0] / nb}, Acc: {i[1] / nb}, F1: {i[2] / nb}, MCC: {i[3] / nb}')
    # return (valid_stats[0][0] + valid_stats[1][0]) / nb

    model.train()
    training_loss = 0
    for batch in tqdm(train_dl, position=0, leave=True):
        opt.zero_grad()

        inputs = batch['seq'].float()
        labels = batch['pumping'].long()

        preds = model(inputs)
        loss = F.cross_entropy(preds, labels)

        training_loss += loss.item()

        loss.backward()
        opt.step()

    nb = len(train_dl)
    print(f"Training loss: {training_loss/nb}")
