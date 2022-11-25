# %%
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import scipy.signal
import torch
import torch.nn as nn
import torchsummary
import heapq
from sklearn.svm import SVR
import deepdish as dd
from sklearn import preprocessing 

# %%
#inisialize
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("cpu")

print(f'Devices: {device}')

# %%
#read file and load models
h5path = '/Users/mclinwong/GitHub/CodesReproduction/DCN-DOA/Data/h5/'
pthpath = '/Users/mclinwong/GitHub/CodesReproduction/DCN-DOA/Data/pth/'

class DNN_ReLU(nn.Module):
    def __init__(self):
        super(DNN_ReLU, self).__init__()
        self.fc1 = nn.Linear(2*L, int(2*L/3))
        self.fc2 = nn.Linear(int(2*L/3), int(4*L/9))
        self.fc3 = nn.Linear(int(4*L/9), int(2*L/3))
        self.fc4 = nn.Linear(int(2*L/3), L)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        return x
dnnrelu = torch.load(pthpath + 'dnnrelu.pth') 

# %%
#read mat file
matpath = '/Users/mclinwong/GitHub/CodesReproduction/DCN-DOA/Data/matlib/'
read_temp=scipy.io.loadmat(matpath + 'data2_test.mat')
S_est=read_temp['S_est']
S_label=read_temp['S_label']
R_est=read_temp['R_est']
S_est = S_est.transpose(0, 2, 1)

DOA_train=read_temp['DOA_train']
theta=read_temp['theta']
gamma=read_temp['gamma']
gamma_R=read_temp['gamma_R']

S_label1 = np.expand_dims(S_label, 2)
S_label1 = S_label1.transpose(0, 2, 1)
normalizer = preprocessing.Normalizer().fit(R_est)
[r2,I,c] = np.shape(S_est)
[r2,c]=np.shape(R_est)
[r2,I]=np.shape(S_label)
print(f'Test Numbers={r2}\nC=M*(M-1)={c}\nDOA Range={I}')

# %%
#Testing
dnnrelu_predict = np.zeros((r2,I))

model = dnnrelu.to(device)
model.eval()

for i in range(r2):
    with torch.no_grad():
        x = torch.from_numpy(S_est[i,:,:]).float().to(device)
        x = x.to(device)
        y = model(x)
        dnnrelu_predict[i,:] = y.cpu().numpy()


        
    


# %%



