from utils import *
from LoadData import *

label = LoadData(dataset_library=dataset_library, filename=train_filename).LABEL()
trainset_fcn = LoadData(dataset_library=dataset_library, filename=train_filename).DATA_EXPAND()
trainset_cnn = LoadData(dataset_library=dataset_library, filename=train_filename).DATA_ORIGINAL()

print('---------------TRAINING SET---------------')
print('trainset_fcn.shape: ', trainset_fcn.shape)
print('trainset_cnn.shape: ', trainset_cnn.shape)
print('label.shape:        ', label.shape)
print('------------------------------------------')

