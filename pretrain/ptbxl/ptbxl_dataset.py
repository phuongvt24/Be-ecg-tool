import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os

class PTBXL(Dataset):
    CLASSES = ['1dAVb', 'RBBB', 'LBBB', 'SB', 'ST', 'AF', 'normal_ecg']
    
    def __init__(self, purpose, data_x=None, data_y=None, data_dir='./data'):
        if data_x is not None and data_y is not None:
            self.x = data_x
            self.Y = data_y[['ecg_id'] + self.CLASSES]
        else:
            assert purpose in ['train', 'val', 'test']
            x_dir = os.path.join(data_dir, f'x_{purpose}.csv')
            y_dir = os.path.join(data_dir, f'y_{purpose}.csv')
            
            self.x = pd.read_csv(x_dir)
            self.Y = pd.read_csv(y_dir)[['ecg_id'] + self.CLASSES]
        
        self.y = self.Y[self.CLASSES].values
        
    def __len__(self):
        return self.Y.shape[0]
    
    def __getitem__(self, index: int):
        ecg_id = self.Y['ecg_id'].iloc[index]
        # labels: One-hot values
        labels = torch.tensor(
            self.Y[self.CLASSES].iloc[index].values
        )
        
        # signals: 2-D array of size (4096, 12) representing 12-lead signals of length 4096
        signals = self.x[self.x.ecg_id == ecg_id].drop(columns=['ecg_id']).values
        
        signals = torch.tensor(signals)
        
        return (signals, labels)


class PTBXL2(Dataset):
    CLASSES = ['1dAVb', 'RBBB', 'LBBB', 'SB', 'ST', 'AF', 'normal_ecg']
    
    def __init__(self, purpose, data_x=None, data_y=None, data_dir='./data'):
        if data_x is not None:
            print('vào none 1')
            self.x = data_x
            if data_y is not None:
                print('vào không none 1')
                self.Y = data_y[['ecg_id'] + self.CLASSES]
                self.Y = self.Y[self.CLASSES].values
        else:
            assert purpose in ['train', 'val', 'test']
            x_dir = os.path.join(data_dir, f'x_{purpose}.csv')
            y_dir = os.path.join(data_dir, f'y_{purpose}.csv')
            
            self.x = pd.read_csv(x_dir)
            if os.path.exists(y_dir):
                self.Y = pd.read_csv(y_dir)[['ecg_id'] + self.CLASSES]
                self.Y = self.Y[self.CLASSES].values
    
    def __len__(self):
        if hasattr(self, 'x') and hasattr(self, 'Y'):
            return self.Y.shape[0]
        
        elif hasattr(self, 'x') and not hasattr(self, 'Y'):
            print('vào none 2')
            return 1

    def __getitem__(self, index: int):
        if hasattr(self, 'x') and hasattr(self, 'Y'):
            # ecg_id = self.Y['ecg_id'].iloc[0]
            # labels: One-hot values
            print('self.Y: ', self.Y)
            labels = torch.tensor(
                self.Y
            )
            
            # signals: 2-D array of size (4096, 12) representing 12-lead signals of length 4096
            signals = self.x.drop(columns=['ecg_id']).values
            
            signals = torch.tensor(signals)
            
            return (signals, labels)
        
        elif hasattr(self, 'x') and not hasattr(self, 'Y'):
            print('vào none 3')
            signals = self.x.drop(columns=['ecg_id']).values
            signals = torch.tensor(signals)
            return signals