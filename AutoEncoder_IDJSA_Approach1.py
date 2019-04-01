#!/usr/bin/env python
# coding: utf-8

# In[9]:


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



# In[11]:


def align(coordinate_file):
    '''
    Input: File contains lines, where each line contains the coordinates of a model, e.g., if model 1 has 70 atoms, each with 3 coordinates  (3*70 = 210 coordinates),
    then the line corresponding model 1 is like this:  210 x1 y1 z1 x2 y2 z2 ... x70 y70 z70

    Alignes all the model with the first model in the cordinate_file.

    Returns: a dictionary of aligned models. Each model, i.e., each entry (value) in the dictionary is a flattened numpy array.

    NOTE: For my leader cluster codes, do not flatten the arrays.
    '''

    modelDict = {}
    ind = 0
    ref = []
    sup = SVDSuperimposer()
    with open(coordinate_file) as f:
        for line in f:
            if ind == 0:
                l = [float(t) for t in line.split()]
                l = l[1:]  # 1ail:l[1:211]
                samples = [l[i:i + 3] for i in range(0, len(l), 3)]
                ref = array(samples, 'f')

                modelDict[ind] = np.ravel(ref)
                ind += 1
            else:
                l = [float(t) for t in line.split()]
                l = l[1:]  # 1ail:l[1:211]
                samples = [l[i:i + 3] for i in range(0, len(l), 3)]
                seq = array(samples, 'f')
                s = sup.set(ref, seq)
                sup.run()
                z = sup.get_transformed()
                modelDict[ind] = np.ravel(z)
                ind += 1
    return modelDict, ref


# In[68]:



coordinate_file = 'onedtja.txt'
#print(coordinate_file)
# align the models with the first one in the file. No need to call this function if already pickled to disk.

aligned_dict, ref = align(coordinate_file)
#print(aligned_dict)
print(ref.shape)


# In[73]:


centered_data, mean = pca.center(aligned_dict)

#print(centered_data.shape[0])
#print(centered_data.shape[1])
array = np.arange(centered_data.shape[0]*centered_data.shape[1]).reshape(centered_data.shape[1], centered_data.shape[0]) 
#print(array.shape)
X = array[0:1000,:]
#X = centered_data[0:1000,:]
#print(X.shape[0]) #Rows
#print(X.shape[1]) #Columns
train, test = train_test_split(X,train_size=0.6, test_size=0.4, random_state=42)
print(train.shape)
print(test.shape)
#XTrain = np.array(centered_data[:,0:600])
#XTest = np.array(centered_data[:,600:1000])
#XTest = np.array(centered_data[:,600:1000])
#XTrain = np.array(centered_data[:,0:600])

#XTest = np.array(centered_data[:,600:1000])
#print(XTrain.shape)
#print(XTest.shape)


# In[74]:



#train, test = train_test_split(X,train_size=0.6, test_size=0.4, random_state=42)
train_x = np.array(train)
test_x = np.array(test)

## we ignore the flatten value needed only when pixel data???
#train_x = train_x.reshape((len(train_x), np.prod(train_x.shape[1:])))
#test_x = test_x.reshape((len(test_x), np.prod(test_x.shape[1:])))

#print(train_x.shape)
#print(test_x.shape)


# In[75]:


x_train = train_x.astype('float32') / 255.
#print(x_train)
x_test = test_x.astype('float32') / 255.
#print(x_test)

## we ignore the flatten value needed only when pixel data???
#x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
#x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
#print(x_train.shape)
#print(x_test.shape)


# # Shapes of training set
# print("Training set  shape: {shape}".format(shape=x_train.shape))

# # Shapes of test set
# print("Test set shape: {shape}".format(shape=x_test.shape))





#different aproach from https://www.kaggle.com/saivarunk/dimensionality-reduction-using-keras-auto-encoder

# define the number of features
#ncol = X_scaled  #Need to check again
#input_dim4 = Input(shape = (ncol, ))
#input_dim4=X_scaled.shape[1]
input_img = Input(shape=(222,))
encoding_dim = 2  
#input_img3 = Input(shape=(input_dim4,))
# Encoder Layers
encoded1 = Dense(250, activation = 'relu')(input_img)
encoded2 = Dense(150, activation = 'relu')(encoded1)
encoded3 = Dense(75, activation = 'relu')(encoded2)
encoded4 = Dense(encoding_dim, activation = 'relu')(encoded3)


# Decoder Layers
decoded1 = Dense(75, activation = 'relu')(encoded4)
decoded2 = Dense(150, activation = 'relu')(decoded1)
decoded3 = Dense(250, activation = 'relu')(decoded2)
decoded4 = Dense(ncol, activation = 'sigmoid')(decoded3)



# Combine Encoder and Deocder layers
autoencoder = Model(inputs = input_img, outputs = decoded4)

# Compile the Model
autoencoder.compile(optimizer = 'adadelta', loss = 'binary_crossentropy')


# In[100]:


autoencoder.summary()


# In[88]:


history = autoencoder.fit(x_train, x_train,
                epochs=10,
                batch_size=16,
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


# In[101]:


history = autoencoder.fit(x_train, x_train,
                epochs=1000,
                batch_size=32,
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


# In[91]:


#history= autoencoder.fit(x_train, x_train, nb_epoch = 100, batch_size = 32, shuffle = False, validation_split=0.1)


#plot our loss 
#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
#plt.title('model train vs validation loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.legend(['train', 'validation'], loc='upper right')
#plt.show()


# In[107]:


#Separate Encoder Model
encoder=Model(input_dim,encoded4)

#Separate Decoder Model
encoded_input= Input(shape=(encoding_dim,))
decoder_layer = autoencoder.layers[-1]
decoder=Model(encoded_input,decoder_layer(encoded_input))

#####
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)


# In[108]:


#Use Encoder level to reduce dimension of train and test data
encoder = Model(inputs = input_dim, outputs = encoded4)
encoded_input = Input(shape = (encoding_dim, ))


# In[113]:


#Predict the new train and test data using Encoder
encoded_train = encoder.predict(train_scaled)


encoded_test = (encoder.predict(test_scaled))


# Plot
plt.plot(encoded_train[:,:])
plt.show()

# Plot
plt.plot(encoded_test[:,:])
plt.show()


# In[114]:


encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

# display reconstruction
plt.plot(decoded_imgs[:,:])
plt.show()

# display original
plt.plot(x_test[:,:])
plt.show()

#Use Encoder level to reduce dimension of train and test data
encoder = Model(inputs = input_dim, outputs = encoded5)
encoded_input = Input(shape = (encoding_dim, ))
print(encoded_input[0:10])


# In[63]:


a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
print(a.shape[0])
print(a.shape[1])
train, test = train_test_split(X,train_size=0.6, test_size=0.4, random_state=42)
print(train.shape)
print(test.shape)


# In[ ]:




