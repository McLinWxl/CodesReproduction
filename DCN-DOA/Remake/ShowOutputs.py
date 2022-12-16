import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import scipy.signal
import torch
import torch.nn as nn
import heapq
from sklearn.svm import SVR
import deepdish as dd
from sklearn import preprocessing 

from ultis import *
from InitData import *
from DefineFunctions import *

# Load model
CNN_ReLU = torch.load(pthpath + 'CNN_ReLU.pth')
DNN_ReLU = torch.load(pthpath + 'DNN_ReLU.pth')

# Test Spectrum
S_est, S_abs, S_label, R_est, S_label1, DOA_train, theta, gamma, gamma_R, normalizer = LoadData.read_spectrum(matlib)
# Test
prediction_DNN_ReLU = Test(CNN_ReLU, S_est, 0)
# Plot
plot_spectrum('CNN_ReLU', DOA_train, prediction_DNN_ReLU)