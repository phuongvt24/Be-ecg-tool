from pretrain.eval import classify, get_f1, get_auprc, classify2
from pretrain.load_model import load_model
from pretrain.resnet1d18 import ResNet1d18
from pretrain.CDIL import CDILClassifier 
from pretrain.eff import EfficientNetB0
import numpy as np
import pandas as pd
from pretrain.ptbxl.ptbxl_dataset import PTBXL ,PTBXL2
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
import torch
import warnings

data_dir = 'ptbxl'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict_resnet18(x,y):
    print('gọi resnet18')
    model_instance = './pretrain/ptbxl/checkpoints/resnet1d18_ptbxl_2024-05-02T05_10_46.358852.pt' 
    model = load_model(ResNet1d18, model_instance, device) 

    if x is not None and y is not None:
        test_data = PTBXL('test',x, y, f'{data_dir}/data/')
    else:
        test_data = PTBXL2('test',x, y, f'{data_dir}/data/')

    if x is not None and y is not None:
        results = classify(model, device, test_data, 1)
    else:
        results = classify2(model, device, test_data, 1)

    if isinstance(results, tuple):
        y_trues, y_scores = results
        y_preds = np.round(y_scores) 
        return y_trues[0], y_preds[0]
    else:
        y_scores = results
        y_preds = np.round(y_scores)
        y_trues = [[]]
        return y_trues[0], y_preds[0]

    # print('y_preds: ', y_preds)
    # y_preds = np.round(y_preds) 
    # return(y_trues[0], y_preds[0])

# def predict_resnet18(x, y=None):
#     test_data = PTBXL('test', x, y, f'{data_dir}/data/')
#     model_instance = './pretrain/ptbxl/checkpoints/resnet1d18_ptbxl_2024-05-02T05_10_46.358852.pt' 
#     model = load_model(ResNet1d18, model_instance, device) 
#     results = classify(model, device, test_data, 1)

#     # Kiểm tra kết quả trả về có bao gồm y_trues hay không
#     if isinstance(results, tuple):
#         y_trues, y_scores = results
#         print("True Labels:", y_trues[0])
#         print("Predicted Scores:", y_scores[0])
#         y_preds = np.round(y_scores) 
#         return y_trues[0], y_preds[0]
#     else:
#         y_scores = results
#         print("Predicted Scores:", y_scores[0])
#         y_preds = np.round(y_scores)
#         y_trues = [[]]
#         return y_trues[0], y_preds[0]


def predict_eff(x, y):
    print('gọi eff')
    model_instance = './pretrain/ptbxl/checkpoints/EfficientNetB0_ptbxl_2024-06-05T13_48_19.043503.pt' 
    model = load_model(EfficientNetB0, model_instance, device) 

    if x is not None and y is not None:
        test_data = PTBXL('test',x, y, f'{data_dir}/data/')
    else:
        test_data = PTBXL2('test',x, y, f'{data_dir}/data/')

    if x is not None and y is not None:
        results = classify(model, device, test_data, 1)
    else:
        results = classify2(model, device, test_data, 1)

    if isinstance(results, tuple):
        y_trues, y_scores = results
        y_preds = np.round(y_scores) 
        return y_trues[0], y_preds[0]
    else:
        y_scores = results
        y_preds = np.round(y_scores)
        y_trues = [[]]
        return y_trues[0], y_preds[0]


def predict_cdil(x,y):
    print('gọi cdil')    
    model_instance = './pretrain/ptbxl/checkpoints/CDILClassifier_ptbxl_2024-06-02T16_24_21.062314.pt' 
    model = load_model(CDILClassifier, model_instance, device) 

    if x is not None and y is not None:
        test_data = PTBXL('test',x, y, f'{data_dir}/data/')
    else:
        test_data = PTBXL2('test',x, y, f'{data_dir}/data/')

    if x is not None and y is not None:
        results = classify(model, device, test_data, 1)
    else:
        results = classify2(model, device, test_data, 1)

    if isinstance(results, tuple):
        y_trues, y_scores = results
        y_preds = np.round(y_scores) 
        return y_trues[0], y_preds[0]
    else:
        y_scores = results
        y_preds = np.round(y_scores)
        y_trues = [[]]
        return y_trues[0], y_preds[0]
