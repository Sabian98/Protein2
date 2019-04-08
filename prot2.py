#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pca
import numpy as np
import xml.etree.ElementTree as ET
import Bio.SVDSuperimposer
from Bio.SVDSuperimposer import SVDSuperimposer
from math import sqrt
from numpy import array, dot
import random
import operator
import os
import sys
import pickle
from sklearn.decomposition import PCA
import math
import scipy.io
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense
from keras.models import Model
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.preprocessing import minmax_scale
from numpy.random import seed
from sklearn.preprocessing import StandardScaler


# In[2]:

def data():
	coordinate_file = 'onedtja.txt'
	aligned_dict, ref = pca.align(coordinate_file)
#Read the Energy List

	energylist = []
	#read the energy file
	with open('onedtja_energy.txt', 'r') as f:
		for line in f:
			line = float(line.strip())
			energylist.append(line)          
	energyarray = np.array(energylist).reshape(-1,1)
	# print(energyarray.shape)


# In[4]:


#converting Aligned dictionary to Aligned Array
	alignedlist=[]
	for key, val in aligned_dict.items():
		alignedlist.append(val)
	alignedArray = np.array(alignedlist)
	centered_data, mean = pca.center(alignedArray)
	centered_data=centered_data.T



#print(test_scaled.shape)
	scaler = StandardScaler()
	scaler.fit(centered_data)
	X_scaled = scaler.transform(centered_data)
# print(X_scaled.shape)
# print(X_scaled)



	initialpc_and_energy = np.concatenate((X_scaled, energyarray), axis = 1)



# #Instead of All data, here we are just taking first 1000 data
	X = initialpc_and_energy[0:200,:]
	train, test = train_test_split(X,train_size=0.6, test_size=0.4, random_state=42)


# # separate Energy from train data 
	train_after_enegy, EnergyAfterTrain = train[:, 0:222], train[:, 222:223]


# # separate Energy from test data
	test_after_enegy, EnergyAfterTest = test[:, 0:222], test[:, 222:223]



	train_x = np.array(train_after_enegy)
	test_x = np.array(test_after_enegy)
	return train_x,test_x



# # ####################### APROACH#############################
# # #create an AE and fit it with our data using 3 neurons in the dense layer using keras' functional API
# # # input_dim = X.shape[1]
# encoding_dim = 2  
# input_img = Input(shape=(222,))
 


# encoded = Dense(128, activation = 'linear')(input_img)
# encoded = Dense(64, activation = 'softplus')(encoded)
# encoded = Dense(2, activation = 'softplus')(encoded)

# # #encoded = Dense(128)(input_img)
# # #LR = LeakyReLU(alpha=0.1)(encoded)
# # #encoded = Dense(64)(LR)
# # #LR = LeakyReLU(alpha=0.1)(encoded)
# # #encoded = Dense(2, activation = 'softplus')(LR)
# # #encoded = Dense(2, activation = 'softplus')(LR)



# # Decoder Layers
# # Decoder Layers
# decoded = Dense(64, activation = 'softplus')(encoded)
# decoded = Dense(128, activation = 'softplus')(decoded)
# decoded = Dense(222, activation = 'linear')(decoded)

# # #decoded = Dense(64)(encoded)
# # #LR = LeakyReLU(alpha=0.1)(decoded)
# # #decoded = Dense(128)(LR)
# # #LR = LeakyReLU(alpha=0.1)(decoded)
# # #decoded = Dense(2, activation = 'sigmoid')(LR)


# # # ######################
# autoencoder = Model(input_img, decoded)
# #autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
# autoencoder.compile(optimizer='adam', loss='mse')

# print(autoencoder.summary())


# # # In[148]:


# # #history = autoencoder.fit(train_x, train_x,
# # #                epochs=1000,
# # #                batch_size=100,
# # #                shuffle=True,
# # #                validation_split=0.1,
# # #                verbose = 0)
# # #print(history.history)   


# # # In[21]:


# # #Should we use train_x or x_train ?

# history = autoencoder.fit(train_x, train_x,
                # epochs=2000,
                # batch_size=64,
                # shuffle=True,
                # validation_split=0.1,
                # verbose = 0)
				
# # #plot our loss 
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model train vs validation loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'validation'], loc='upper right')
# plt.show()				
			


# # # In[22]:


# # Use Encoder level to reduce dimension of train and test data
# encoder = Model(input_img, encoded)
# encoded_input = Input(shape=(encoding_dim,))

# # ###### Use decoder level
# # # decoder_layer = autoencoder.layers[-1]
# decoder_layer1 = autoencoder.layers[-3]
# decoder_layer2 = autoencoder.layers[-2]
# decoder_layer3 = autoencoder.layers[-1]
# # # decoder = Model(encoded_input, decoder_layer(encoded_input))
# decoder = Model(input=encoded_input, output=decoder_layer3(decoder_layer2(decoder_layer1(encoded_input))))

# print(decoder.summary())


# # # In[25]:


# # #encoded_data = encoder.predict(test_scaled)
# encoded_data = encoder.predict(test_x)
# print(encoded_data.shape)
# print(encoded_data)

# # decoded_data = decoder.predict(encoded_data)
# # print(decoded_data.shape)
# # print(decoded_data-test_x)

# auto_data=autoencoder.predict(test_x)
# print(auto_data.shape)
# print(test_x-auto_data)
# # with open('Version2decodedData.txt', 'w') as f:
	# # np.savetxt(f, decoded_data, delimiter = ' ', fmt='%1.8f')


# # # In[27]:


# # #Predict the new train and test data using Encoder
# # encoded_train = encoder.predict(train_x)


# # encoded_test = (encoder.predict(test_x))


# # # Plot
# # plt.plot(encoded_train[:,:])
# # plt.show()

# # # Plot
# # plt.plot(encoded_test[:,:])
# # plt.show()


# # # In[28]:


# # #Concat auto encoder result with enegy
# # encoder_and_energy = np.concatenate((encoded_data, EnergyAfterTest), axis = 1)

# # with open('Version2AutoEncoder_energy_1dtja.txt', 'w') as f:
	# # np.savetxt(f, encoder_and_energy, delimiter = ' ', fmt='%1.8f')


# # # In[ ]:




