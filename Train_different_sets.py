
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, explained_variance_score,r2_score
import pickle
import time
import copy
import random

#%% Read and manipulate data

# Settings
pTrain = 0.8
pValidate = 0.1
pTest = 0.1

#Load data from file
df_random = pd.read_csv('resultsIP.csv')
df_uniform = pd.read_csv('FRTData_uni.csv')
df_component = pd.read_csv('FRTData_comp.csv')
df_sys = pd.read_csv('FRTData_sys.csv')

flag = 2 # 1 trained on all data, 2 on random data, and 3 on edge cases

if flag == 1:
        data = np.concatenate((df_random.to_numpy(),df_uniform.to_numpy(),df_component.to_numpy(),df_sys.to_numpy()))
if flag == 2:
        data = df_random.to_numpy()
if flag == 3:
        data = np.concatenate((df_uniform.to_numpy(),df_component.to_numpy(),df_sys.to_numpy()))
# Definition of inputs and outputs
np.random.shuffle(data)

size_data = data.shape
input_data = data[:,:(size_data[1]-1)]
output_data = data[:,-1]
edge_data = np.concatenate((df_uniform.to_numpy(),df_component.to_numpy(),df_sys.to_numpy()))
edge_data_input = edge_data[:,:(size_data[1]-1)]
edge_data_label = edge_data[:,-1]

non_edge_data = df_random.to_numpy()
non_edge_data_input = non_edge_data[:,:(size_data[1]-1)]
non_edge_data_label = non_edge_data[:,-1]

# Normal standarization of data according to

scaler = StandardScaler()
scaler.fit(input_data)

input_data = scaler.transform(input_data)
edge_data_input = scaler.transform(edge_data_input)
non_edge_data_input = scaler.transform(non_edge_data_input)
# Split into training, validation and testing set

p1 = int(size_data[0]*pTrain)
train_data = input_data[:p1,:]
train_label = output_data[:p1]
p2 = int(size_data[0]*(pTrain+pValidate))
valid_data = input_data[p1:p2,:]
valid_label = output_data[p1:p2]
test_data = input_data[p2:,:]
test_label = output_data[p2:]

#Training model

mlp = MLPRegressor(hidden_layer_sizes=20*(150,), learning_rate_init=0.0003, alpha=0.0001)
mlp.fit(train_data, train_label)
print('Model Fitted')

#Saving model for posteriority
#
with open('model_fitted_non_edge.pkl', 'wb') as f:
        pickle.dump(mlp, f)

#Load saved model so we don't need to train everytime

mlp = pickle.load(open('model_fitted_non_edge.pkl','rb'))

# Computing values predicted by the neural network

nn_train_label = mlp.predict(train_data)
nn_test_label = mlp.predict(test_data)

nn_edge_label = mlp.predict(edge_data_input)
nn_non_edge_label = mlp.predict(non_edge_data_input)
#Computing R2 error

r2_train = r2_score(train_label,nn_train_label)
r2_test = r2_score(test_label,nn_test_label)
r2_edge = r2_score(edge_data_label,nn_edge_label)
r2_non_edge = r2_score(non_edge_data_label,nn_non_edge_label)
#plotting scatter plot of NN vs Label



#Printing accuracy values

print('R2 train data')
print(r2_train)
print('R2 test data')
print(r2_test)

print('R2 edge data')
print(r2_edge)

print('R2 non-edge data')
print(r2_non_edge)
