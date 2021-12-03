
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

r_list_train  = []
r_list_test = []

indexes = [0,33,29,7,15,9,16,17,8,32,11]
indexes.reverse()


for i in range(7,len(indexes)):

        index_local = indexes[0:i]
        input_data = data[:,:(size_data[1]-2)]
        for j in range(size_data[1]-3,0,-1):
                if j not in index_local:
                        input_data = np.delete(input_data, j, axis=1)
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
#
# #Saving model for posteriority
# #
        with open('model_fitted_'+str(i)+'added_.pkl', 'wb') as f:
                pickle.dump(mlp, f)

        #Load saved model so we don't need to train everytime

        #mlp = pickle.load(open('model_fitted_'+str(i)+'_.pkl','rb'))

# Computing values predicted by the neural network

        nn_train_label = mlp.predict(train_data)
        nn_test_label = mlp.predict(test_data)

#Computing R2 error

        r2_train = r2_score(train_label,nn_train_label)
        r2_test = r2_score(test_label,nn_test_label)
        print(r2_test)
        r_list_test.append(r2_test)
        r_list_train.append(r2_train)
        print(i)

with open('OAT/r_test_adding.pkl', 'wb') as f:
        pickle.dump(r_list_test, f)

with open('OAT/r_train_adding.pkl', 'wb') as f:
        pickle.dump(r_list_train, f)

#plotting scatter plot of NN vs Label

# plt.scatter(train_label,nn_train_label,s=4)
# plt.plot([0,40],[0,40],color = 'orange')
# plt.xlabel('Train data')
# plt.ylabel('NN prediction')
# plt.show()

#Printing accuracy values





# ''' Example discussed while teaching how to use the code'''
#
# input_to_evaluate = np.ones((1,41))
# input_to_evaluate[4] = 2.5
# input_to_evaluate = scaler.transform(input_to_evaluate)
# output_value = mlp.predict(input_to_evaluate)



