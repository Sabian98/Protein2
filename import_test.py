
import pca
import numpy as np
import xml.etree.ElementTree as ET

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



coordinate_file = 'F:\GMU\Spring 19\CS 701\protein\onedtja.txt'
aligned_dict, ref = pca.align(coordinate_file)
centered_data, mean = pca.center(aligned_dict)



array = np.arange(centered_data.shape[0]*centered_data.shape[1]).reshape(centered_data.shape[1], centered_data.shape[0]) 
X = array[0:1000,:]
print(X.shape)


train, test = train_test_split(X,train_size=0.6, test_size=0.4, random_state=42)
train_x = np.array(train)
test_x = np.array(test)
train_x = train_x.astype('float32') / 255.
# # #print(x_train)
test_x = test_x.astype('float32') / 255.
# # #print(x_test)

#create an AE and fit it with our data using 3 neurons in the dense layer using keras' functional API
# input_dim = X.shape[1]
encoding_dim = 2  
input_img = Input(shape=(222,))
 


encoded = Dense(128, LeakyReLU(alpha=0.3))(input_img)
encoded = Dense(64, LeakyReLU(alpha=0.3))(encoded)
encoded = Dense(2, LeakyReLU(alpha=0.3))(encoded)



# Decoder Layers
decoded = Dense(64, LeakyReLU(alpha=0.3))(encoded)
decoded = Dense(128, LeakyReLU(alpha=0.3))(decoded)
decoded = Dense(222, activation = 'linear')(decoded)

# ######################
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
print(autoencoder.summary())

history = autoencoder.fit(train_x, train_x,
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
				
# encoder = Model(input_img, encoded)		

# encoded_input = Input(shape=(encoding_dim,))

# decoder_layer = autoencoder.layers[-1]
# # create the decoder model
# decoder = Model(encoded_input, decoder_layer(encoded_input))		

encoder = Model(input_img, encoded)
encoded_input = Input(shape=(encoding_dim,))
# decoder_layer = autoencoder.layers[-1]
decoder_layer1 = autoencoder.layers[-3]
decoder_layer2 = autoencoder.layers[-2]
decoder_layer3 = autoencoder.layers[-1]
# decoder = Model(encoded_input, decoder_layer(encoded_input))
decoder = Model(input=encoded_input, output=decoder_layer3(decoder_layer2(decoder_layer1(encoded_input))))
#decoder=Model(encoded_input,decoder_layer3(encoded_input))
encoded_data = encoder.predict(test_x)
print(encoded_data[:,:2])
				
		














		
				
				
				
				