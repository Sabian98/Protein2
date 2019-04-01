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
from keras.layers.advanced_activations import LeakyReLU
from sklearn.preprocessing import StandardScaler


# In[2]:


coordinate_file = 'onedtja.txt'
#print(coordinate_file)
# align the models with the first one in the file. No need to call this function if already pickled to disk.

aligned_dict, ref = pca.align(coordinate_file)
#print(aligned_dict)
# print(len(aligned_dict))

# with open("myfile.txt", 'w') as f:
    # for key, value in aligned_dict.items():
        # f.write('%s:%s\n' % (key, value))


# In[13]:


#centered_data, mean = pca.center(aligned_dict)

#print(centered_data.shape[0])
#print(centered_data.shape[1])
#array = np.arange(centered_data.shape[0]*centered_data.shape[1]).reshape(centered_data.shape[1], centered_data.shape[0]) 
#print(array.shape)
#X = array[0:1000,:]
#AllCenteredData = array
#print(X.shape[0]) #Rows
#print(X.shape[1]) #Columns
#print(AllCenteredData.shape)


# In[3]:


#Read the Energy List

energylist = []
	#read the energy file
with open('onedtja_energy.txt', 'r') as f:
		for line in f:
			line = float(line.strip())
			energylist.append(line)          
energyarray = np.array(energylist).reshape(-1,1)
print(energyarray.shape)


# In[4]:


#converting Aligned dictionary to Aligned Array
alignedlist=[]
for key, val in aligned_dict.items():
		alignedlist.append(val)
alignedArray = np.array(alignedlist)
# print(alignedArray)
# with open('AlighnedArray.txt', 'w') as f:
	# np.savetxt(f, alignedArray, delimiter = ' ', fmt='%1.8f')
#---------------------------------------------------------------------

def center(data):
	'''
	input: a dictionary where each value is a model in the form of a flattened array, and each array contains the coordinates of the atoms of that model.

	Method:
		Constructs an m by n array where m is the total number of coorniates of all atoms (e.g., for 1ail with 70 atoms,  m = 70 * 3 = 270), and n is the number of models, i.e., n= 50,000+
		subtracts the mean of the row elements from each value of the rows

	returns: the centered array, i.e., the result of the above method
	'''
#Directly taking array now after training
	#biglist = []
	#for key, val in aligned_dict.items():
	#	biglist.append(val)
	#data = np.array(biglist)
	data = data.T
	mean = data.mean(axis=1).reshape(-1, 1)
	data = data - data.mean(axis=1).reshape(-1, 1)
	return data, mean




centered_data, mean = center(alignedArray)
centered_data=centered_data.T

print(centered_data.shape)


# with open('Version2centerData.txt', 'w') as f:
	# np.savetxt(f, centered_data, delimiter = ' ', fmt='%1.8f')


# In[12]:


#Minmax scalled but why?
# train_test_all_scaled = minmax_scale(centered_data, axis = 0)


#print(test_scaled.shape)
scaler = StandardScaler()
scaler.fit(centered_data)
X_scaled = scaler.transform(centered_data)
print(X_scaled.shape)
print(X_scaled)

#ncol = train_scaled.shape[1]
#print(ncol)


#print(test_scaled)


# In[14]:


# #Initial Concat of 1dtja set ( Which is aligned) and their co-ordinated energy

initialpc_and_energy = np.concatenate((X_scaled, energyarray), axis = 1)

# with open('Version2Initial_center_data_energy_1dtja.txt', 'w') as f:
	# np.savetxt(f, initialpc_and_energy, delimiter = ' ', fmt='%1.8f')


# # In[15]:


# print(initialpc_and_energy.shape)


# # In[16]:


# #Instead of All data, here we are just taking first 1000 data
X = initialpc_and_energy[0:200,:]
print(X.shape)


# # In[17]:


train, test = train_test_split(X,train_size=0.6, test_size=0.4, random_state=42)
print(train.shape)
print(test.shape)


# # In[18]:


# # separate Energy from train data 
train_after_enegy, EnergyAfterTrain = train[:, 0:222], train[:, 222:223]
print(train_after_enegy.shape)
print(EnergyAfterTrain.shape)

# # separate Energy from test data
test_after_enegy, EnergyAfterTest = test[:, 0:222], test[:, 222:223]
print(test_after_enegy.shape)
print(EnergyAfterTest.shape)



train_x = np.array(train_after_enegy)
test_x = np.array(test_after_enegy)
print('test data is')
print(test_x)



# ####################### APROACH#############################
# #create an AE and fit it with our data using 3 neurons in the dense layer using keras' functional API
# # input_dim = X.shape[1]
encoding_dim = 2  
input_img = Input(shape=(222,))
 


encoded = Dense(128, activation = 'relu')(input_img)
encoded = Dense(64, activation = 'relu')(encoded)
encoded = Dense(2, activation = 'relu')(encoded)

# #encoded = Dense(128)(input_img)
# #LR = LeakyReLU(alpha=0.1)(encoded)
# #encoded = Dense(64)(LR)
# #LR = LeakyReLU(alpha=0.1)(encoded)
# #encoded = Dense(2, activation = 'softplus')(LR)
# #encoded = Dense(2, activation = 'softplus')(LR)



# Decoder Layers
# Decoder Layers
decoded = Dense(64, activation = 'relu')(encoded)
decoded = Dense(128, activation = 'relu')(decoded)
decoded = Dense(222, activation = 'sigmoid')(decoded)

# #decoded = Dense(64)(encoded)
# #LR = LeakyReLU(alpha=0.1)(decoded)
# #decoded = Dense(128)(LR)
# #LR = LeakyReLU(alpha=0.1)(decoded)
# #decoded = Dense(2, activation = 'sigmoid')(LR)


# # ######################
autoencoder = Model(input_img, decoded)
#autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
autoencoder.compile(optimizer='adam', loss='mse')

print(autoencoder.summary())


# # In[148]:


# #history = autoencoder.fit(train_x, train_x,
# #                epochs=1000,
# #                batch_size=100,
# #                shuffle=True,
# #                validation_split=0.1,
# #                verbose = 0)
# #print(history.history)   


# # In[21]:


# #Should we use train_x or x_train ?

history = autoencoder.fit(train_x, train_x,
                epochs=2000,
                batch_size=20,
                shuffle=True,
                validation_split=0.1,
                verbose = 0)
				
# #plot our loss 
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model train vs validation loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()				
			


# # In[22]:


# Use Encoder level to reduce dimension of train and test data
encoder = Model(input_img, encoded)
encoded_input = Input(shape=(encoding_dim,))

# ###### Use decoder level
# # decoder_layer = autoencoder.layers[-1]
decoder_layer1 = autoencoder.layers[-3]
decoder_layer2 = autoencoder.layers[-2]
decoder_layer3 = autoencoder.layers[-1]
# # decoder = Model(encoded_input, decoder_layer(encoded_input))
decoder = Model(input=encoded_input, output=decoder_layer3(decoder_layer2(decoder_layer1(encoded_input))))

print(decoder.summary())


# # In[25]:


# #encoded_data = encoder.predict(test_scaled)
encoded_data = encoder.predict(test_x)
print(encoded_data.shape)
print(encoded_data)

decoded_data = decoder.predict(encoded_data)
print(decoded_data.shape)
print(decoded_data-test_x)


# with open('Version2decodedData.txt', 'w') as f:
	# np.savetxt(f, decoded_data, delimiter = ' ', fmt='%1.8f')


# # In[27]:


# #Predict the new train and test data using Encoder
# encoded_train = encoder.predict(train_x)


# encoded_test = (encoder.predict(test_x))


# # Plot
# plt.plot(encoded_train[:,:])
# plt.show()

# # Plot
# plt.plot(encoded_test[:,:])
# plt.show()


# # In[28]:


# #Concat auto encoder result with enegy
# encoder_and_energy = np.concatenate((encoded_data, EnergyAfterTest), axis = 1)

# with open('Version2AutoEncoder_energy_1dtja.txt', 'w') as f:
	# np.savetxt(f, encoder_and_energy, delimiter = ' ', fmt='%1.8f')


# # In[ ]:




