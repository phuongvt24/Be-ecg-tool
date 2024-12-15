from pretrain.eval import classify, classify2
from pretrain.load_model import load_model
import torch
import numpy as np
from sklearn.metrics import classification_report
from pretrain.resnet1d18 import ResNet1d18
from pretrain.CDIL import CDILClassifier 
from pretrain.eff import EfficientNetB0
from pretrain.ptbxl.ptbxl_dataset import PTBXL, PTBXL2
import torch.nn as nn
import pandas as pd

class SelfAttention1D(nn.Module):
    def __init__(self, in_features):
        super(SelfAttention1D, self).__init__()
        self.query = nn.Linear(in_features, in_features // 8)
        self.key = nn.Linear(in_features, in_features // 8)
        self.value = nn.Linear(in_features, in_features)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C = x.size()
        proj_query = self.query(x).view(batch_size, -1, C // 8)
        proj_key = self.key(x).view(batch_size, -1, C // 8).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        attention = torch.nn.functional.softmax(energy, dim=-1)
        proj_value = self.value(x).view(batch_size, -1, C).permute(0, 2, 1)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C)
        out = self.gamma * out + x
        return out


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

best_model_path = './pretrain/ptbxl/checkpoints/best_meta_model.pt'

model_checkpoint_1 = './pretrain/ptbxl/checkpoints/CDILClassifier_ptbxl_2024-06-02T16_24_21.062314.pt'
model_checkpoint_2 = './pretrain/ptbxl/checkpoints/resnet1d18_ptbxl_2024-05-02T05_10_46.358852.pt'
model_checkpoint_3 = './pretrain/ptbxl/checkpoints/EfficientNetB0_ptbxl_2024-06-05T13_48_19.043503.pt'

data_dir = 'ptbxl'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict_fusion_att(x,y):
    print('gọi att')
    model1 = load_model(CDILClassifier, model_checkpoint_1, device)
    model2 = load_model(ResNet1d18, model_checkpoint_2, device)
    model3 = load_model(EfficientNetB0, model_checkpoint_3, device)

    if x is not None and y is not None:
        test_data = PTBXL('test',x, y, f'{data_dir}/data/')
    else:
        test_data = PTBXL2('test',x, y, f'{data_dir}/data/')

    if x is not None and y is not None:
        test_y_trues, test_probs1 = classify(model1, device, test_data, 1)
        test_y_trues, test_probs2 = classify(model2, device, test_data, 1)
        test_y_trues, test_probs3 = classify(model3, device, test_data, 1)
    else:
        test_probs1 = classify2(model1, device, test_data, 1)
        test_probs2 = classify2(model2, device, test_data, 1)
        test_probs3 = classify2(model3, device, test_data, 1)

    test_meta_features = np.hstack([test_probs1, test_probs2, test_probs3])

    input_size = test_meta_features.shape[1]
    meta_model_nn = MetaNN(input_size, 7).to(device)

    meta_model_nn.load_state_dict(torch.load(best_model_path, map_location=device))

    meta_model_nn.eval()

    with torch.no_grad():
        inputs_test = torch.tensor(test_meta_features, dtype=torch.float32).to(device)
        outputs_test = meta_model_nn(inputs_test)
        meta_probs = torch.sigmoid(outputs_test).cpu().numpy()
        
    threshold = 0.5
    final_preds = (meta_probs >= threshold).astype(int)
    if x is not None and y is not None:
        return(test_y_trues[0], final_preds[0])
    else:
        return([], final_preds[0])