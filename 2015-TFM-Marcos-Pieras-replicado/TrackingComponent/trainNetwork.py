from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D,GlobalAveragePooling2D
from keras.utils import np_utils
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import h5py
from keras import regularizers
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger

from keras.initializers import Constant
from keras.optimizers import Adam


with h5py.File('/home/marc/Escriptori/siameseDATASET/FirstApproach/train.h5','r') as hdf:

    ls = list(hdf.keys())
    print(ls)
    x_trainR = hdf.get('X')
    y_trainR = hdf.get('y')

    x_train = np.copy(x_trainR)
    y_train = np.copy(y_trainR)
    


mean1 = 118.268211396 
mean2 = 110.23197584
mean3 = 109.777103553
mean4 = 118.316815951
mean5 = 110.257208327
mean6 = 109.849493294

x_train[:,:,:,0] = x_train[:,:,:,0]-mean1
x_train[:,:,:,1] = x_train[:,:,:,1]-mean2
x_train[:,:,:,2] = x_train[:,:,:,2]-mean3
x_train[:,:,:,3] = x_train[:,:,:,3]-mean4
x_train[:,:,:,4] = x_train[:,:,:,4]-mean5
x_train[:,:,:,5] = x_train[:,:,:,5]-mean6


x_train = x_train.astype("float32")
x_train /= 255 
    


with h5py.File('/home/marc/Escriptori/siameseDATASET/FirstApproach/test.h5','r') as hdf:

    ls = list(hdf.keys())
    print(ls)

    x_testR = hdf.get('X')
    y_testR = hdf.get('y')

    x_test = np.copy(x_testR)
    y_test = np.copy(y_testR)


x_test[:,:,:,0] = x_test[:,:,:,0]-mean1
x_test[:,:,:,1] = x_test[:,:,:,1]-mean2
x_test[:,:,:,2] = x_test[:,:,:,2]-mean3
x_test[:,:,:,3] = x_test[:,:,:,3]-mean4
x_test[:,:,:,4] = x_test[:,:,:,4]-mean5
x_test[:,:,:,5] = x_test[:,:,:,5]-mean6


x_test = x_test.astype("float32")
x_test /= 255 



print(np.shape(x_train),np.shape(y_train),np.shape(x_test),np.shape(y_test))

model = Sequential()


input_shape = (128,64, 6)

model.add(Conv2D(20, (3, 3), input_sshape=input_shape, kernel_initializer='he_uniform',name='conv1',activation = 'relu',bias_initializer= Constant(value=0.1)))
model.add(Conv2D(25, (3, 3),name='conv2',kernel_initializer='he_uniform',activation = 'relu',bias_initializer= Constant(value=0.1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(30, (3, 3),name='conv3',kernel_initializer='he_uniform',activation = 'relu',bias_initializer= Constant(value=0.1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(35, (3, 3),name='conv5',kernel_initializer='he_uniform',activation = 'relu',bias_initializer= Constant(value=0.1)))
model.add(Conv2D(35, (3, 3),name='conv6',kernel_initializer='he_uniform',activation = 'relu',bias_initializer= Constant(value=0.1)))
model.add(Conv2D(35, (3, 3),name='conv7',kernel_initializer='he_uniform',activation = 'relu',bias_initializer= Constant(value=0.1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(GlobalAveragePooling2D())
model.add(Dense(128,kernel_initializer='glorot_uniform',bias_initializer= Constant(value=0.1)))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('sigmoid'))
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.00005)
model.compile(loss='binary_crossentropy',
              optimizer=adam,
              metrics=['binary_accuracy'])

#callbacks

csv_logger = CSVLogger('main1.csv', append=True, separator=';')

filepath="main1weights-improvement-{epoch:02d}.hdf5"
checkpoint = ModelCheckpoint(filepath, period=5, verbose=1, save_best_only=False, mode='max')


history = model.fit(x_train, y_train, batch_size=512,epochs=200, verbose=1,shuffle=True, validation_data=(x_test, y_test),callbacks=[csv_logger,checkpoint])
