import numpy as np
from torch.utils.data import  DataLoader, Dataset
from sklearn.model_selection import train_test_split
import scipy.io
import os
from sklearn import preprocessing
from utils import *

class MakeDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        label = self.label[idx]
        data = self.data[idx]
        return data, label
    
class LoadData():
    def __init__(self, dataset_library, filename, batch_size = 64):
        self.batch_size = batch_size
        self.dataset = scipy.io.loadmat(f'{dataset_library}/{filename}')
    def DOA_train(self):
        return self.dataset['DOA_train']
    def R_TRIANGULAR(self):
        return self.dataset['R_est']
    def DATA_ORIGINAL(self):
        return self.dataset['S_est']
    def DATA_EXPAND(self):
        return self.dataset['S_abs']
    def LABEL(self):
        return self.dataset['S_label']
    def LABEL_EXPAND(self):
        return self.dataset['S_label1']
        
        
print("end")