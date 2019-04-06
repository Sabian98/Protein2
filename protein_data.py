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



#--------------#
coordinate_file = 'onedtja.txt'

# align the models with the first one in the file. No need to call this function if already pickled to disk.

aligned_dict, ref = pca.align(coordinate_file)
#--------------#
energylist = []
	#read the energy file
with open('onedtja_energy.txt', 'r') as f:
		for line in f:
			line = float(line.strip())
			energylist.append(line)          
energyarray = np.array(energylist).reshape(-1,1)
#---------------#

#converting Aligned dictionary to Aligned Array
alignedlist=[]
for key, val in aligned_dict.items():
		alignedlist.append(val)
alignedArray = np.array(alignedlist)

#---------------#
centered_data, mean = pca.center(alignedArray)
centered_data=centered_data.T
scaler = StandardScaler()
scaler.fit(centered_data)
X_scaled = scaler.transform(centered_data)

#---------------#
# #Initial Concat of 1dtja set ( Which is aligned) and their co-ordinated energy

initialpc_and_energy = np.concatenate((X_scaled, energyarray), axis = 1)

train, test = train_test_split(X,train_size=0.6, test_size=0.4, random_state=42)



# # In[18]:


# # separate Energy from train data 
train_after_enegy, EnergyAfterTrain = train[:, 0:222], train[:, 222:223]
# print(train_after_enegy.shape)
# print(EnergyAfterTrain.shape)

# # separate Energy from test data
test_after_enegy, EnergyAfterTest = test[:, 0:222], test[:, 222:223]
# print(test_after_enegy.shape)
# print(EnergyAfterTest.shape)



train_x = np.array(train_after_enegy)
test_x = np.array(test_after_enegy)
return train_x,test_x
#---------------#
#---------------#
#---------------#





































