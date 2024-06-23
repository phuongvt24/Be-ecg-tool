import torch
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, classification_report
from pretrain.ptbxl.ptbxl_dataset import PTBXL  
from pretrain.CDIL import CDILClassifier 
from pretrain.eval import classify, get_f1, get_auprc
from pretrain.resnet1d18 import ResNet1d18
from pretrain.eff import EfficientNetB0

def load_model(model_class, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = model_class(num_classes=len(PTBXL.CLASSES))
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    return model
