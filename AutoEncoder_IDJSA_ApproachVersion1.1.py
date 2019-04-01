#!/usr/bin/env python
# coding: utf-8

# In[2]:


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



# In[12]:


coordinate_file = 'onedtja.txt'
#print(coordinate_file)
# align the models with the first one in the file. No need to call this function if already pickled to disk.

aligned_dict, ref = pca.align(coordinate_file)
#print(aligned_dict)
print(len(aligned_dict))


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


# In[37]:


#Read the Energy List

energylist = []
	#read the energy file
with open('onedtja_energy.txt', 'r') as f:
		for line in f:
			line = float(line.strip())
			energylist.append(line)
#energylist = energylist[:-1]  # only for 1aly            
energyarray = np.array(energylist).reshape(-1,1)
#Xenergyarray = energyarray[0:400,:]
print(energyarray.shape)


# In[18]:


alignedlist=[]
for key, val in aligned_dict.items():
		alignedlist.append(val)
alignedArray = np.array(alignedlist)
print(alignedArray.shape)


# In[21]:


#Initial Concat of 1dtja set ( Which is aligned) and their co-ordinated energy

initialpc_and_energy = np.concatenate((alignedArray, energyarray), axis = 1)

with open('Initial_data_energy_1dtja.txt', 'w') as f:
	np.savetxt(f, initialpc_and_energy, delimiter = ' ', fmt='%1.8f')


# In[22]:


print(initialpc_and_energy.shape)


# In[43]:


#Instead of All data, here we are just taking first 1000 data
X = initialpc_and_energy[0:1000,:]
print(X.shape)


# In[44]:


train, test = train_test_split(X,train_size=0.6, test_size=0.4, random_state=42)
print(train.shape)
print(test.shape)


# In[45]:


# separate Energy from train data 
train_after_enegy, EnergyAfterTrain = train[:, 0:222], train[:, 222:223]
print(train_after_enegy.shape)
print(EnergyAfterTrain.shape)

# separate Energy from test data
test_after_enegy, EnergyAfterTest = test[:, 0:222], test[:, 222:223]
print(test_after_enegy.shape)
print(EnergyAfterTest.shape)


# In[46]:


# Do we need to centerize data ?
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
	data = data - data.mean(axis=1).reshape(-1, 1)#####deducts mean of each column from the column values
	return data, mean


# In[55]:


train_x = np.array(train_after_enegy)
test_x = np.array(test_after_enegy)
print(train_x.shape)
print(test_x.shape)


# In[53]:


# Do we need to centerize data ? 

centered_data_train, mean = center(train_x)
centered_data_train = np.arange(centered_data_train.shape[0]*centered_data_train.shape[1]).reshape(centered_data_train.shape[1], centered_data_train.shape[0]) 

print(centered_data_train.shape)



centered_data_test, mean = center(test_x)
centered_data_test = np.arange(centered_data_test.shape[0]*centered_data_test.shape[1]).reshape(centered_data_test.shape[1], centered_data_test.shape[0]) 

print(centered_data_test.shape)


# In[56]:


#If we need Center data, then we will do this part:

train_x=centered_data_train
test_x=centered_data_test
print(train_x.shape)
print(test_x.shape)


# In[59]:


#Convert all the values between 0 to 1

x_train = train_x.astype('float32') / 255.
x_test = test_x.astype('float32') / 255.

# Shapes of training set
print("Training set  shape: {shape}".format(shape=x_train.shape))

# Shapes of test set
print("Test set shape: {shape}".format(shape=x_test.shape))


# In[61]:


#Minmax scaled but why?
train_scaled = minmax_scale(x_train, axis = 0)
test_scaled = minmax_scale(x_test, axis = 0)
print(train_scaled.shape)
print(test_scaled.shape)

ncol = train_scaled.shape[1]
print(ncol)


# In[62]:


####################### APROACH#############################
#create an AE and fit it with our data using 3 neurons in the dense layer using keras' functional API
# input_dim = X.shape[1]
encoding_dim = 2  
input_img = Input(shape=(222,))
 


encoded = Dense(128, activation = 'softplus')(input_img)
encoded = Dense(64, activation = 'softplus')(encoded)
encoded = Dense(2, activation = 'softplus')(encoded)



# Decoder Layers
decoded = Dense(64, activation = 'softplus')(encoded)
decoded = Dense(128, activation = 'softplus')(decoded)
decoded = Dense(222, activation = 'sigmoid')(decoded)

# ######################
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
print(autoencoder.summary())


# In[75]:


#Should we use train_x or x_train ?

history = autoencoder.fit(train_x, train_x,
                epochs=1000,
                batch_size=100,
                shuffle=True,
                validation_split=0.1,
                verbose = 0)
				
				
#plot our loss 
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model train vs validation loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()				
			


# In[69]:


# Write some comments here
encoder = Model(input_img, encoded)
encoded_input = Input(shape=(encoding_dim,))
# decoder_layer = autoencoder.layers[-1]
decoder_layer1 = autoencoder.layers[-3]
decoder_layer2 = autoencoder.layers[-2]
decoder_layer3 = autoencoder.layers[-1]
# decoder = Model(encoded_input, decoder_layer(encoded_input))
decoder = Model(input=encoded_input, 
output=decoder_layer3(decoder_layer2(decoder_layer1(encoded_input))))
# encoded_data = encoder.predict(X_scaled)
#encoded_data = encoder.predict(test_scaled)
#print(encoded_data)


# In[71]:


encoded_data = encoder.predict(test_scaled)
print(encoded_data.shape)
print(encoded_data)


# In[72]:


#Predict the new train and test data using Encoder
encoded_train = encoder.predict(train_scaled)


encoded_test = (encoder.predict(test_scaled))


# Plot
plt.plot(encoded_train[:,:])
plt.show()

# Plot
plt.plot(encoded_test[:,:])
plt.show()


# In[74]:


#Concat auto encoder result with enegy
encoder_and_energy = np.concatenate((encoded_data, EnergyAfterTest), axis = 1)

with open('AutoEncoder_energy_1dtja.txt', 'w') as f:
	np.savetxt(f, encoder_and_energy, delimiter = ' ', fmt='%1.8f')


# In[ ]:




