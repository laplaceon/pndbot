import torch
import torch.nn.functional as F

import numpy as np

from torch.optim import Adam

from data import get_data_loaders
from model import PumpDiscriminator, PndModel

from tqdm import tqdm
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

def compute_accuracy(pred, labels):
    pred = np.argmax(pred, axis=1)

    return {
        "acc": accuracy_score(labels, pred),
        "f1": f1_score(labels, pred),
        "roc": roc_auc_score(labels, pred)
    }

def train_loop(model, opt, train_dl, val_dl, epochs):
    for i in range(epochs):
        model.eval()
        valid_stats = 1 * [[0, 0, 0, 0]]
        for batch in tqdm(val_dl, position=0, leave=True):
            with torch.no_grad():
                inputs = batch['seq'].float().cuda()
                next = batch['next'].float().cuda()
                labels = batch['pumping'].long().cuda()

                preds = model(inputs, next)
                # print(labels)

                loss_0 = F.cross_entropy(preds[0], labels)

                scores = compute_accuracy(preds[0].detach().cpu().numpy(), labels.detach().cpu().numpy())

                valid_stats[0] = [a + b for a, b in zip(valid_stats[0], [loss_0.item(), scores["acc"], scores["f1"], scores["roc"]])]
                # valid_stats[1] = [a + b for a, b in zip(valid_stats[1], [0, 0, 0, 0])]
        nb = len(val_dl)
        for i in valid_stats:
            print(f'Valid loss: {i[0] / nb}, Acc: {i[1] / nb}, F1: {i[2] / nb}, MCC: {i[3] / nb}')
        # return (valid_stats[0][0] + valid_stats[1][0]) / nb

        model.train()
        training_loss = 0
        for batch in tqdm(train_dl, position=0, leave=True):
            opt.zero_grad()

            inputs = batch['seq'].float().cuda()
            next = batch['next'].float().cuda()
            labels = batch['pumping'].long().cuda()

            preds = model(inputs, next)
            loss = F.cross_entropy(preds[0], labels) + F.mse_loss(preds[1].permute(1, 0, 2), next)

            training_loss += loss.item()

            loss.backward()
            opt.step()

        nb = len(train_dl)
        print(f"Training loss: {training_loss/nb}")

train_dl, val_dl = get_data_loaders('./data/chart_landmarks.csv', './data/charts/', bs=128, multiplier=256)
model = PndModel()
model.cuda()
opt = Adam(model.parameters(), lr=1e-4)

train_loop(model, opt, train_dl, val_dl, 200)
