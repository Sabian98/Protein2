#use S P A C E S you halfwit


from keras.models import Sequential, Model
from keras.layers import Dense, Input, Flatten, Reshape
from keras.datasets import mnist
from keras.optimizers import Adam,SGD
from keras.initializers import RandomNormal
import numpy as np
import matplotlib
import helpers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pca


matplotlib.use('Agg')

import matplotlib.pyplot as plt
plt.ioff()

initializer = RandomNormal(mean=0.0, stddev=0.01, seed=None)
class AAN():
    def __init__(self, img_shape=(222), encoded_dim=2):
        self.encoded_dim = encoded_dim
        self.optimizer_reconst = Adam(0.01)
        self.optimizer_discriminator = Adam(0.01)
        self._initAndCompileFullModel(img_shape, encoded_dim)

    def _genEncoderModel(self, img_shape, encoded_dim):
        """ Build Encoder Model Based on Paper Configuration
        Args:
            img_shape (tuple) : shape of input image
            encoded_dim (int) : number of latent variables
        Return:
            A sequential keras model
        """
        encoder = Sequential()
        #encoder.add(Flatten(input_shape=img_shape))
        encoder.add(Dense(1000, activation='relu', kernel_initializer=initializer,
                bias_initializer=initializer))
        encoder.add(Dense(1000, activation='relu', kernel_initializer=initializer,
                bias_initializer=initializer))
        encoder.add(Dense(encoded_dim, kernel_initializer=initializer,
                bias_initializer=initializer))
        # encoder.summary()
        return encoder

    def _getDecoderModel(self, encoded_dim, img_shape):
        """ Build Decoder Model Based on Paper Configuration
        Args:
            encoded_dim (int) : number of latent variables
            img_shape (tuple) : shape of target images
        Return:
            A sequential keras model
        """
        decoder = Sequential()
        decoder.add(Dense(1000, activation='relu', input_dim=encoded_dim, kernel_initializer=initializer,
                bias_initializer=initializer))
        decoder.add(Dense(1000, activation='relu', kernel_initializer=initializer,
                bias_initializer=initializer))
        decoder.add(Dense(np.prod(img_shape), activation='sigmoid', kernel_initializer=initializer,
                bias_initializer=initializer))
        # decoder.add(Reshape(img_shape))
        # decoder.summary()
        return decoder
		
    def _getDescriminator(self, encoded_dim):
        """ Build Descriminator Model Based on Paper Configuration
        Args:
            encoded_dim (int) : number of latent variables
        Return:
            A sequential keras model
        """
        discriminator = Sequential()
        discriminator.add(Dense(1000, activation='relu',
                                input_dim=encoded_dim, kernel_initializer=initializer,
                bias_initializer=initializer))
        discriminator.add(Dense(1000, activation='relu', kernel_initializer=initializer,
                bias_initializer=initializer))
        discriminator.add(Dense(1, activation='sigmoid', kernel_initializer=initializer,
                bias_initializer=initializer))
        # discriminator.summary()
        return discriminator
    
	
	
    def _initAndCompileFullModel(self, img_shape, encoded_dim):
        self.encoder = self._genEncoderModel(img_shape, encoded_dim)
        self.decoder = self._getDecoderModel(encoded_dim, img_shape)
        self.discriminator = self._getDescriminator(encoded_dim)
        img = Input(shape=(img_shape,))#err
        encoded_repr = self.encoder(img)
        gen_img = self.decoder(encoded_repr)
        self.autoencoder = Model(img, gen_img)
        valid = self.discriminator(encoded_repr)
        self.encoder_discriminator = Model(img, valid)
        self.discriminator.compile(optimizer=self.optimizer_discriminator,
                                   loss='binary_crossentropy',
                                   metrics=['accuracy'])
        self.autoencoder.compile(optimizer=self.optimizer_reconst,
                                 loss ='mse')
        for layer in self.discriminator.layers:
            layer.trainable = False
        self.encoder_discriminator.compile(optimizer=self.optimizer_discriminator,
                                           loss='binary_crossentropy',
                                           metrics=['accuracy'])
    def generateImages(self, n=100):
        latents = 5*np.random.normal(size=(n, self.encoded_dim))
        imgs = self.decoder.predict(latents)
        return imgs
    def train(self, x_train, batch_size=100, epochs=1000, save_interval=500):
        half_batch = int(batch_size / 2)
        for epoch in range(epochs):
            #---------------Train Discriminator -------------
            # Select a random half batch of images
            idx = np.random.randint(0, x_train.shape[0], half_batch)
            imgs = x_train[idx]
            # Generate a half batch of new images
            latent_fake = self.encoder.predict(imgs)
            #gen_imgs = self.decoder.predict(latent_fake)
            latent_real = 5*np.random.normal(size=(half_batch, self.encoded_dim))
            valid = np.ones((half_batch, 1))
            fake = np.zeros((half_batch, 1))
            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(latent_real, valid)
            d_loss_fake = self.discriminator.train_on_batch(latent_fake, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            idx = np.random.randint(0, x_train.shape[0], batch_size)
            imgs = x_train[idx]
            # Generator wants the discriminator to label the generated representations as valid
            valid_y = np.ones((batch_size, 1))

            # Train the autoencode reconstruction
            g_loss_reconstruction = self.autoencoder.train_on_batch(imgs, imgs)

            # Train generator
            g_logg_similarity = self.encoder_discriminator.train_on_batch(imgs, valid_y)
            # Plot the progress
            print ("%d [D loss: %f, acc: %.2f%%] [G acc: %f, mse: %f]" % (epoch, d_loss[0], 100*d_loss[1],
                   g_logg_similarity[1], g_loss_reconstruction))
            # if(epoch % save_interval == 0):
                # self.imagegrid(epoch)


if __name__ == '__main__':
    # Load MNIST dataset
    # (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # x_train = x_train.astype(np.float32) / 255.
    # x_test = x_test.astype(np.float32) / 255.
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
    X = initialpc_and_energy[0:500,:]

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

	
    ann = AAN(encoded_dim=2)
    ann.train(train_x)
    generated = ann.generateImages(10000)
    L= helpers.approximateLogLiklihood(generated, test_x)
    # print "Log Likelihood"
    # print L












