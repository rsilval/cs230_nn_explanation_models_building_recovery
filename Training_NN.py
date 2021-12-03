
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
df = pd.read_csv('resultsIP.csv')
data = df.to_numpy()

# Definition of inputs and outputs

size_data = data.shape
input_data = data[:,:(size_data[1]-2)]
output_data = data[:,-1]

# Normal standarization of data according to

scaler = StandardScaler()
scaler.fit(input_data)
input_data = scaler.transform(input_data)

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
with open('model_fitted.pkl', 'wb') as f:
        pickle.dump(mlp, f)

#Load saved model so we don't need to train everytime

mlp = pickle.load(open('model_fitted.pkl','rb'))

# Computing values predicted by the neural network

nn_train_label = mlp.predict(train_data)
nn_test_label = mlp.predict(test_data)

#Computing R2 error

r2_train = r2_score(train_label,nn_train_label)
r2_test = r2_score(test_label,nn_test_label)

#plotting scatter plot of NN vs Label

# plt.scatter(train_label,nn_train_label,s=4)
# plt.plot([0,40],[0,40],color = 'orange')
# plt.xlabel('Train data')
# plt.ylabel('NN prediction')
# plt.show()

#Printing accuracy values

print('R2 train data')
print(r2_train)
print('R2 test data')
print(r2_test)

'''Perform sensitivity analysis'''
n_component = 41
n_increases = 21

t0 = time.time()

sens_matrix = np.zeros((n_component,n_increases))

# ''' Example discussed while teaching how to use the code'''
#
# input_to_evaluate = np.ones((1,41))
# input_to_evaluate[4] = 2.5
# input_to_evaluate = scaler.transform(input_to_evaluate)
# output_value = mlp.predict(input_to_evaluate)



for i in range(n_component):
        for j in range(n_increases):
                vector_input = np.ones((1,n_component))
                vector_input[0][i] = 1 + j*0.1
                vector_input = scaler.transform(vector_input)
                pred_val = mlp.predict(vector_input)
                sens_matrix[i][j] = pred_val[0]
t1 = time.time()

print(t1-t0)

with open('sensitivity_matrix.pkl','wb') as f:
        pickle.dump(sens_matrix,f)

np.savetxt('sensitivity_matrix.csv',sens_matrix,delimiter=',')

vector = np.reshape(sens_matrix,-1)

vector_altered = copy.deepcopy(vector)
#