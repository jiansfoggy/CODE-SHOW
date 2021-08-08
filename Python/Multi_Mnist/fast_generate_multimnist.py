import os,sys,json,time,math,glob,random,keras
import numpy as np
from numpy import random
import tensorflow as tf
from keras.utils import np_utils

# Initialize variables that are not initialized yet
def batch_deformation(batch_images, max_shift = 2, keep_dim= True):
    batch_size, h, w, c = batch_images.shape #(batch_size, 28, 28, 1)
    deform_batch = np.zeros([batch_size, h+2*max_shift, w+2*max_shift, c])
    for idx in range(batch_size):
        off_set = np.random.randint(0, 2*max_shift + 1, 2)
        deform_batch[idx, off_set[0]:off_set[0]+h, off_set[1]:off_set[1]+w, :] = batch_images[idx]
        off_set = None
        del off_set
    if keep_dim:
        return deform_batch[:,max_shift:max_shift+h, max_shift: max_shift+w, :] 
    else:
        return deform_batch

def multi_batch(batch_x1, batch_y1, batch_x2, batch_y2):
    batch_x1 = batch_deformation(batch_x1, max_shift=4, keep_dim=False)
    batch_x2 = batch_deformation(batch_x2, max_shift=4, keep_dim=False)
    batch_x = (batch_x1 + batch_x2)

    batch_y = np.clip(batch_y1+batch_y2, 0, 1)
  
    return batch_x, batch_y

# download MNIST dataset from keras
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# convert data type to float 32
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.

# Create and Prepare dataset
X1 = []
Y1 = []
x_ind=np.concatenate((x_train,x_test),axis=0)
y_ind=np.concatenate((y_train,y_test),axis=0)
x_train, y_train, x_test, y_test = None, None, None, None
del x_train
del y_train
del x_test
del y_test

repeat_time=10
totl_smpl=repeat_time*70000
trn_smpl =repeat_time*60000
for i in range(repeat_time):
    if len(X1)==0:
        X1 = x_ind
        Y1 = y_ind
    else:
        X1 = np.concatenate((X1,x_ind),axis=0)
        Y1 = np.concatenate((Y1,y_ind),axis=0)
print("Image Shape is:", np.shape(X1))
print("Label Shape is:", np.shape(Y1))
# Start Shuffling and Combine
rsort = list(range(totl_smpl))
np.random.shuffle(rsort)
X2 = X1[rsort,:,:,:]            
Y2 = Y1[rsort]

batch_y1 = np_utils.to_categorical(Y1, 10)
batch_y2 = np_utils.to_categorical(Y2, 10)
y1_train = np.vstack([Y1[:trn_smpl], Y2[:trn_smpl]]).T
y1_test  = np.vstack([Y1[trn_smpl:], Y2[trn_smpl:]]).T
Y1, Y2   = None, None
del Y1
del Y2
batch_x, batch_y, X1, X2 = multi_batch(X1, batch_y1, X2, batch_y2)
batch_y1, batch_y2 = None, None
del batch_y1
del batch_y2

x_train  = batch_x[:trn_smpl,:,:,:]
x_test   = batch_x[trn_smpl:,:,:,:]
batch_x  = None
del batch_x
x1_train = X1[:trn_smpl,:,:,:]
x2_train = X2[:trn_smpl,:,:,:]
x1_test  = X1[trn_smpl:,:,:,:]
x2_test  = X2[trn_smpl:,:,:,:]
X1, X2   = None, None
del X1
del X2

y_train  = batch_y[:trn_smpl,:]
y_test   = batch_y[trn_smpl:,:]
batch_y  = None
del batch_y

print("Job Finished.")
print("x_train shape: ", np.shape(x_train))
print("y_train shape: ", np.shape(y_train))
print("x_test  shape: ", np.shape(x_test))
print("y_test  shape: ", np.shape(y_test))
