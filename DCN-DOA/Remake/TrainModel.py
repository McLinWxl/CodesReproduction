import torch
import numpy as np

from ultis import *
from InitData import *
from DefineNetwork import *
from DefineFunctions import *

#fit Seed
myseed = 8974  # set a random seed for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)

#make directory
make_dir(matlib, pthpath, resultpath, figpath)

#read data
S_est, S_abs, S_label, R_est, S_label1, Sample, L, dim = LoadData.read_data(matlib)
#data loader
train_loader, valid_loader, train_loader_fcn, valid_loader_fcn = LoadData.data_loader(S_est, S_abs, S_label, S_label1, batch_size)

#train model 
CNN_ReLU = CNN(nn.ReLU).to(device)
DNN_ReLU = DNN(nn.ReLU).to(device)

#Fully-Connected Network
DNN_train, DNN_valid = train_save(train_loader_fcn, valid_loader_fcn, DNN_ReLU=DNN_ReLU)

#Convolutional Neural Network
CNN_train, CNN_valid = train_save(train_loader, valid_loader, CNN_ReLU=CNN_ReLU)

#plot training and validation loss
plot_loss('Train', DNN_ReLU=models[1], CNN_ReLU=models[0])
plot_loss('Valid', DNN_ReLU=models[1], CNN_ReLU=models[0])

