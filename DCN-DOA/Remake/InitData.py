import numpy as np
from torch.utils.data import  DataLoader, Dataset
from sklearn.model_selection import train_test_split
import scipy.io
import os
from sklearn import preprocessing
from ultis import *

def make_dir(*args):
    for value in args:
        if not os.path.exists(value):
            os.makedirs(value)

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
    def __init__(self, matlib, batch_size):
        self.matlib = matlib
        self.batch_size = batch_size
    def read_data(matlib):
        read_data = scipy.io.loadmat(matlib + 'data2_trainlow.mat')
        S_est = read_data['S_est']
        S_abs = read_data['S_abs']
        S_label = read_data['S_label']
        R_est = read_data['R_est']
        S_label1 = np.expand_dims(S_label, 2)
        [Sample, L, dim] = np.shape(S_est)
        S_est = S_est.transpose(0, 2, 1)
        S_label1 = S_label1.transpose(0, 2, 1)
        print('----------Shape of suorce data:----------')
        print(f'S_est.shape: {S_est.shape}')
        print(f'S_abs.shape: {S_abs.shape}')
        print(f'S_label.shape: {S_label.shape}')
        print(f'S_label1.shape: {S_label1.shape}')
        print(f'Sample: {Sample}, L: {L}, dim: {dim}')
        return S_est, S_abs, S_label, R_est, S_label1, Sample, L, dim
    def read_spectrum(matlib):
        read_temp=scipy.io.loadmat(matlib + 'data2_test.mat')
        S_est=read_temp['S_est']
        S_est = S_est.transpose(0, 2, 1)
        [r2,K,I] = np.shape(S_est)
        S_real = np.zeros((r2, 1, I))
        S_imag = np.zeros((r2, 1, I))
        S_real[:,0,:] = S_est[:,0,:]
        S_imag[:,0,:] = S_est[:,1,:]
        S_abs = np.append(S_real, S_imag, axis=2)
        S_abs = np.squeeze(S_abs)
        S_label=read_temp['S_label']
        R_est=read_temp['R_est']
        DOA_train=read_temp['DOA_train']
        theta=read_temp['theta']
        gamma=read_temp['gamma']
        gamma_R=read_temp['gamma_R']
        S_label1 = np.expand_dims(S_label, 2)
        normalizer = preprocessing.Normalizer().fit(R_est)
        [r2,c]=np.shape(R_est)
        [r2,I]=np.shape(S_label)
        print('----------Shape of spectrum data:----------')
        print(f'r2: {r2}, I: {I}, c: {c}')
        print(f'S_est.shape: {np.shape(S_est)}')
        print(f'S_abs.shape: {np.shape(S_abs)}')
        return S_est, S_abs, S_label, R_est, S_label1, DOA_train, theta, gamma, gamma_R, normalizer
    def data_loader(S_est, S_abs, S_label, S_label1, batch_size):
        S_est_train, S_est_test, S_label1_train, S_label1_test = train_test_split(S_est, S_label1, test_size=0.2)
        S_abs_train, S_abs_test, S_label_train, S_label_test = train_test_split(S_abs, S_label, test_size=0.2)
        print('----------Shape of devided data:----------')
        print(f'S_est_train.shape: {S_est_train.shape}, S_est_valid.shape: {S_est_test.shape}')
        print(f'S_abs_train.shape: {S_abs_train.shape}, S_abs_valid.shape: {S_abs_test.shape}')
        print(f'S_label1_train.shape: {S_label1_train.shape}, S_label1_valid.shape: {S_label1_test.shape}')

        train_set = MakeDataset(S_est_train, S_label1_train)
        train_set_fcn = MakeDataset(S_abs_train, S_label_train)
        valid_set = MakeDataset(S_est_test, S_label1_test)
        valid_set_fcn = MakeDataset(S_abs_test, S_label_test)

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)
        train_loader_fcn = DataLoader(train_set_fcn, batch_size=batch_size, shuffle=True)
        valid_loader_fcn = DataLoader(valid_set_fcn, batch_size=batch_size, shuffle=False)
        
        return train_loader, valid_loader, train_loader_fcn, valid_loader_fcn