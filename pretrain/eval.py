import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, auc, f1_score

def classify(model, device, dataset, epoch, batch_size=128):
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    y_trues = np.empty((0, len(dataset.CLASSES)))
    y_scores = np.empty((0, len(dataset.CLASSES)))
    incorrect_predictions = []

    model.eval()
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(data_loader):
            X, y = X.to(device), y.to(device)
            X = X.permute(0, 2, 1)
            y_hat = model(X.float())
            
            y_scores_batch = torch.sigmoid(y_hat).cpu().numpy()
            y = y.cpu().numpy()

            y_scores = np.concatenate((y_scores, y_scores_batch), axis=0)
            y_trues = np.concatenate((y_trues, y), axis=0)
            
            y_preds_batch = np.round(y_scores_batch)
            incorrect_indices = np.where(np.any(y != y_preds_batch, axis=1))[0]
            for idx in incorrect_indices:
                incorrect_predictions.append({
                    'ecg_id': dataset.Y['ecg_id'].iloc[batch_idx * batch_size + idx],
                    'true_labels': y[idx],
                    'predicted_labels': y_preds_batch[idx]
                })

    return y_trues, y_scores

def get_f1(y_trues, y_preds):
    f1 = []
    for j in range(y_trues.shape[1]):
        f1.append(f1_score(y_trues[:, j], y_preds[:, j]))
    return np.array(f1)

def get_auprc(y_trues, y_scores):
    auprc = []
    for j in range(y_trues.shape[1]):
        p, r, thresholds = precision_recall_curve(y_trues[:, j], y_scores[:, j])
        auprc.append(auc(r, p))

    return np.array(auprc)

