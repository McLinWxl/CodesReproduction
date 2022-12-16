import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import torch
import torchvision.models as models
import torch.nn as nn
from torch.utils.data import  DataLoader, Dataset
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import scipy.io
import os
import heapq
from ultis import *

def train(model,name , train_loader, valid_loader, epoch):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    train_loss_list = []
    valid_loss_list = []
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = []
        
        for x, y in tqdm(train_loader):
            optimizer.zero_grad()
            x = x.to('cpu').float()
            x = x.to(device)
            y = y.to('cpu').float()
            y = y.to(device)
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            
        train_loss = np.mean(train_loss)
        train_loss_list.append(train_loss)
        print('Epoch: {}, Train Loss: {:.4f}'.format(epoch, train_loss))
    
        model.eval()
        valid_loss = []
        
        for x, y in tqdm(valid_loader):
            x = x.to('cpu').float()
            x = x.to(device)
            y = y.to('cpu').float()
            y = y.to(device)
            with torch.no_grad():
                output = model(x)
            loss = criterion(output, y)
            valid_loss.append(loss.item())
            
        valid_loss = np.mean(valid_loss)
        valid_loss_list.append(valid_loss)
        print('Epoch: {}, Valid Loss: {:.4f}'.format(epoch, valid_loss))
    # save loss as csv
    id = np.arange(0, num_epochs)
    datafarme = pd.DataFrame({'id':id ,'train_loss':train_loss_list, 'valid_loss':valid_loss_list})
    datafarme.to_csv(resultpath + name +'loss.csv', index=False, sep=',')
    return train_loss_list, valid_loss_list

def Test(model, inputx, flag=1):
    model = model.to(device)
    model.eval()
    ls = []
    for i in range(r2):
        with torch.no_grad():
            if flag == 0:
                x = np.zeros((K, I))
                x[:, :] = inputx[i, :, :]
                #atttntion: the input of the model should be a tensor which is in the shape of (batch_size, channel, length)
                x = torch.from_numpy(x.reshape(1, K, I)).float().to('cpu')
            elif flag == 1:
                x = np.zeros(K*I)
                x[:] = inputx[i, :]
                x = torch.from_numpy(x.reshape(1, -1)).float().to('cpu')
            x = x.to(device)
            # print(np.shape(x))
            y = model(x)
            y = y.cpu().numpy()
            ls.append(y)
    predict = np.array(ls)
    return np.squeeze(predict)

def train_save(train_loader, valid_loader, **kargs):
    num_model = len(kargs)
    for name, model in kargs.items():
        i = 0
        train_list = np.zeros((num_epochs, num_model))
        valid_list = np.zeros((num_epochs, num_model))
        train_list[:, i], valid_list[:, i] = train(model, name, train_loader, valid_loader, num_epochs)
        torch.save(model, pthpath + name + '.pth')
        i += 1
    return train_list, valid_list

def plot_loss(tltle, **kwargs):
    with plt.style.context(['science']):
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['savefig.dpi'] = 300
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.title(tltle)
        for name, list in kwargs.items():
            plt.plot(list, label=str(name))
        plt.legend()
        plt.savefig(figpath + tltle + '.pdf')
        plt.show()

def DOAPredict(predict, height = 0.1, nodetect = 0):
    peak = np.zeros((K, r2))
    for i in range(r2):
        li = predict[i,:]
        peaks_st = np.zeros((K))
        peaks_st = peaks_st + nodetect
        peaks,_ = scipy.signal.find_peaks(li, height=height)
        maxamp = heapq.nlargest(K, li[peaks])
        rank = np.zeros(np.shape(maxamp)[0])
        for s in range(np.shape(maxamp)[0]):
            rank[s] = np.where(li==maxamp[s])[0].item()
        
        if len(peaks) == K:
            peaks_st = peaks
        elif len(peaks) == 0:
            peaks_st = peaks_st
        elif len(peaks) < K:
            for t in range(len(peaks)):
                peaks_st[t] = peaks[t]
        elif len(peaks) > K:
            for j in range(K):
                peaks_st[j] = rank[j]

        peak[:,i] = sorted(peaks_st, reverse=True)

    return peak-60

def load_model(*args):
    for model in args:
        return torch.load(pthpath + model + '.pth')        
        