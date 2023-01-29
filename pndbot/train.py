import torch
import torch.nn.functional as F

import numpy as np

from torch.optim import Adam

from data import get_data_loaders
import model

from pytorchtools import EarlyStopping

from tqdm import tqdm
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

def compute_accuracy(pred, labels):
    pred = np.argmax(pred, axis=1)

    return {
        "acc": accuracy_score(labels, pred),
        "f1": f1_score(labels, pred),
        "roc": 0
    }

def eval(model, val_dl):
    model.eval()
    valid_stats = 2 * [[0, 0, 0, 0]]
    for batch in tqdm(val_dl, position=0, leave=True):
        with torch.no_grad():
            inputs = batch['seq'].float().cuda()
            # next = batch['next'].float().cuda()
            labels = batch['pumping'].long().cuda()

            preds = model(inputs)

            loss_0 = F.cross_entropy(preds, labels)
            # loss_1 = F.mse_loss(preds[1], next)

            scores = compute_accuracy(preds.detach().cpu().numpy(), labels.detach().cpu().numpy())

            valid_stats[0] = [a + b for a, b in zip(valid_stats[0], [loss_0.item(), scores["acc"], scores["f1"], scores["roc"]])]
            valid_stats[1] = [a + b for a, b in zip(valid_stats[1], [0, 0, 0, 0])]
    nb = len(val_dl)
    for i in valid_stats:
        print(f'Valid loss: {i[0] / nb}, Acc: {i[1] / nb}, F1: {i[2] / nb}, MCC: {i[3] / nb}')
    return (valid_stats[0][0] + valid_stats[1][0]) / nb

def train_loop(model, opt, train_dl, val_dl, chk):
    early_stopping = EarlyStopping(patience=10, verbose=True, path=chk)
    
    epoch = 0

    while True:
        epoch += 1
        print(f"Epoch {epoch}")

        val_loss = eval(model, val_dl)

        model.train()
        training_loss = 0
        for batch in tqdm(train_dl, position=0, leave=True):
            opt.zero_grad()

            inputs = batch['seq'].float().cuda()
            # next = batch['next'].float().cuda()
            labels = batch['pumping'].long().cuda()

            # print(inputs)

            preds = model(inputs)
            loss_0 = F.cross_entropy(preds, labels)
            # loss_1 = F.mse_loss(preds[1], next)

            loss = loss_0
            # loss = 0.7 * loss_0 + 0.3 * loss_1

            training_loss += loss.item()

            loss.backward()
            opt.step()

        nb = len(train_dl)
        print(f"Training loss: {training_loss/nb}")

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

train_dl, val_dl = get_data_loaders('../data/chart_landmarks_auto_overlapped.csv', '../data/full/binance/', bs=128, multiplier=192)
model = model.PndModel()
model.cuda()
opt = Adam(model.parameters(), lr=5e-5)

train_loop(model, opt, train_dl, val_dl, '../models/clstm.pt')
