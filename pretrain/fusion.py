from eval import classify, get_f1, get_auprc
from test import load_model
import torch
from resnet1d18 import ResNet1d18
from CDIL import CDILClassifier 
from eff import EfficientNetB0
import numpy as np
import pandas as pd
from ptbxl.ptbxl_dataset import PTBXL 
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, f1_score
import copy
import os
import torch.nn as nn
import torch.optim as optim
import json


class MetaNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(MetaNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model_checkpoint_1 = './ptbxl/checkpoints/CDILClassifier_ptbxl_2024-06-02T16_24_21.062314.pt'  
model_checkpoint_2 = './ptbxl/checkpoints/resnet1d18_ptbxl_2024-05-02T05_10_46.358852.pt'  
model_checkpoint_3 = './ptbxl/checkpoints/EfficientNetB0_ptbxl_2024-06-05T13_48_19.043503.pt'  

data_dir = 'ptbxl'
batch_size = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_data = PTBXL('train', f'{data_dir}/data/')
test_data = PTBXL('test', f'{data_dir}/data/')

# LOAD MODELS
model1 = load_model(CDILClassifier, model_checkpoint_1, device)
model2 = load_model(ResNet1d18, model_checkpoint_2, device)
model3 = load_model(EfficientNetB0, model_checkpoint_3, device)

train_y_trues, train_probs1 = classify(model1, device, train_data, 1)
train_y_trues, train_probs2 = classify(model2, device, train_data, 1)
train_y_trues, train_probs3 = classify(model3, device, train_data, 1)

test_y_trues, test_probs1 = classify(model1, device, test_data, 1)
test_y_trues, test_probs2 = classify(model2, device, test_data, 1)
test_y_trues, test_probs3 = classify(model3, device, test_data, 1)

train_meta_features = np.hstack([train_probs1, train_probs2, train_probs3])
test_meta_features = np.hstack([test_probs1, test_probs2, test_probs3])

input_size = train_meta_features.shape[1]
output_size = train_y_trues.shape[1]
meta_model_nn = MetaNN(input_size, output_size).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(meta_model_nn.parameters(), lr=0.001)
num_epochs = 100 

best_f1 = 0
best_model_weights = copy.deepcopy(meta_model_nn.state_dict())
best_model_path = './ptbxl/checkpoints/best_meta_model.pt'

os.makedirs(os.path.dirname(best_model_path), exist_ok=True)

for epoch in range(num_epochs):
    meta_model_nn.train()
    optimizer.zero_grad()

    inputs = torch.tensor(train_meta_features, dtype=torch.float32).to(device)
    labels = torch.tensor(train_y_trues, dtype=torch.float32).to(device)

    # Forward pass
    outputs = meta_model_nn(inputs)
    loss = criterion(outputs, labels)

    # Backward pass and update
    loss.backward()
    optimizer.step()

    meta_model_nn.eval()
    with torch.no_grad():
        inputs_val = torch.tensor(test_meta_features, dtype=torch.float32).to(device)
        outputs_val = meta_model_nn(inputs_val)
        outputs_val = torch.sigmoid(outputs_val).cpu().numpy()
        preds_val = (outputs_val >= 0.5).astype(int)

    current_f1 = f1_score(test_y_trues, preds_val, average='macro')

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Validation F1: {current_f1:.4f}')

    if current_f1 > best_f1:
        best_f1 = current_f1
        best_model_weights = copy.deepcopy(meta_model_nn.state_dict())
        torch.save(meta_model_nn.state_dict(), best_model_path)
        print(f'Saved best model to {best_model_path} at epoch {epoch+1} with F1-Score: {best_f1:.4f}')

meta_model_nn.load_state_dict(best_model_weights)
print(f'Best F1-Score: {best_f1:.4f}')

meta_model_nn.eval()
with torch.no_grad():
    inputs_test = torch.tensor(test_meta_features, dtype=torch.float32).to(device)
    outputs_test = meta_model_nn(inputs_test)
    meta_probs = torch.sigmoid(outputs_test).cpu().numpy()

threshold = 0.5
final_preds = (meta_probs >= threshold).astype(int)

print(classification_report(test_y_trues, final_preds))
