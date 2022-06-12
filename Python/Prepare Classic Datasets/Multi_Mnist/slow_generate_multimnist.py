import os, sys,time, math, glob, joblib
import numpy as np
from numpy import random
import tensorflow as tf
from operator import add
import keras.backend as K
from keras.utils import np_utils
from joblib import Parallel, delayed
from contextlib import contextmanager

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print('{0} done in {1:.3f} seconds.'.format(name, time.time() - t0))

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
  batch_x1, batch_x2 = None, None
  del batch_x2
  del batch_x1
  return batch_x, batch_y

def outer_batch(X1,Y1,X2,Y2, x_train,y_train,x_test,y_test, cls_num, i,j):
    batch_x1 = X1[i]
    batch_y1 = np_utils.to_categorical(Y1[i], 10)
    # j is the class index waiting for fused into X1 and Y1
    it=0  
    x_multi, y_multi = [], []
    ind_ls = np.random.choice(cls_num[j],size=(2,cls_num[i]), replace=True)
    #ind_ls = np.random.randint(cls_num[j],size=(2,cls_num[i]))
    while it<2:     
        #ind_ls = np.random.randint(cls_num[j],size=cls_num[i])                 
        batch_x2 = X2[j][ind_ls[it,:],:,:,:]            
        batch_y2 = Y2[j][ind_ls[it]]            
        batch_y2 = np_utils.to_categorical(batch_y2, 10)
        batch_x, batch_y = multi_batch(batch_x1, batch_y1, batch_x2, batch_y2)
        if len(x_multi)==0: 
            x_multi=batch_x
            y_multi=batch_y
        else:
            x_multi=np.concatenate((x_multi,batch_x), axis=0)
            y_multi=np.concatenate((y_multi,batch_y), axis=0)
        it+=1
        batch_x2, batch_y2, batch_x, batch_y = None, None, None, None           
        del batch_x2
        del batch_y2
        del batch_x
        del batch_y
    print("finished inner inner 2 round")
    ind_ls = None
    del ind_ls

    ind_test = np.random.choice(cls_num[i]*2,size=2000, replace=False)
    #ind_test = np.random.randint(cls_num[i]*2,size=2000)
    def delete_ele(x_multi, y_multi, k):
        x_multi.pop(k)
        y_multi.pop(k)
        return x_multi, y_multi

    if len(x_train)==0: 
        x_test = x_multi[ind_test,:,:,:]
        y_test = y_multi[ind_test,:]
        x_multi = x_multi.tolist()
        y_multi = y_multi.tolist()
        for ele in sorted(ind_test, reverse = True):
            x_multi.pop(ele)
            y_multi.pop(ele)
        x_train = x_multi
        y_train = y_multi
        print("finished split test and train set at ",j,"round")
    else:
        x_test = np.concatenate((x_test,x_multi[ind_test,:,:,:]), axis=0)
        y_test = np.concatenate((y_test,y_multi[ind_test,:]), axis=0)
        x_multi = x_multi.tolist()
        y_multi = y_multi.tolist()
        for ele in sorted(ind_test, reverse = True):
            x_multi.pop(ele)
            y_multi.pop(ele)
        x_train = np.concatenate((x_train,x_multi), axis=0)
        y_train = np.concatenate((y_train,y_multi), axis=0)
        print("finished split test and train set at ",j,"round")
    ind_test, x_multi, y_multi  = None, None, None
    del ind_test
    del x_multi
    del y_multi 
    print("inner",j,"chunk is finished.")

    print("outer",i,"chunk is finished.")
    batch_x1, batch_y1 = None, None
    del batch_x1
    del batch_y1
    return x_train,y_train,x_test,y_test

# download MNIST dataset from keras
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# convert data type to float 32
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.

#num_lst_train = []
#num_lst_test = []
New_X = []
New_Y = []
cls_num_train = []
cls_num_test = []
for i in range(10):
    # prepare index and count number
    cls_ind_train = np.where(y_train==i)[0]
    cls_num_train.append(len(cls_ind_train))
    #num_lst_train.append(cls_ind_train)
    cls_ind_test = np.where(y_test==i)[0]
    cls_num_test.append(len(cls_ind_test))
    #num_lst_test.append(cls_ind_test)
    # extract value and assemble train and test data
    x_ind=np.concatenate((x_train[cls_ind_train,:,:,:],x_test[cls_ind_test,:,:,:]),axis=0)
    y_ind=np.concatenate((y_train[cls_ind_train],y_test[cls_ind_test]),axis=0)
    New_X.append(x_ind)
    New_Y.append(y_ind)
    cls_ind_train, cls_ind_test = None, None
    del cls_ind_train 
    del cls_ind_test
    #New_Y.append(list(y_ind))

# class number for each category and present class number
cls_num = list(map(add, cls_num_train, cls_num_test))
print("class number is :",*cls_num)
# generate two back up variables
X1, X2 = New_X, New_X
Y1, Y2 = New_Y, New_Y
New_X, New_Y = None, None
del New_X
del New_Y

# prepare image and label
x_train, y_train, x_test, y_test = [], [], [], []
with timer('joblib parallel datetime processing threads:'):
    output = Parallel(n_jobs=5, prefer='threads')(delayed(outer_batch)(X1,Y1,X2,Y2, 
        x_train,y_train,x_test,y_test, cls_num, i,j) 
        for i in range(10) for j in range(10))
q_ind = 0
for q in output:
    if q_ind==0:
        x_train = q[0]
        y_train = q[1]
        x_test  = q[2]
        y_test  = q[3]
        q_ind   = 1
    else:
        x_train = np.concatenate((x_train,q[0]), axis=0)
        y_train = np.concatenate((y_train,q[1]), axis=0)
        x_test  = np.concatenate((x_test, q[2]), axis=0)
        y_test  = np.concatenate((y_test, q[3]), axis=0)

for i in range(10):
    batch_x1 = X1[i]
    batch_y1 = Y1[i]
    batch_y1 = np_utils.to_categorical(batch_y1, 10)
    for j in range(10):
        it=0  
        x_multi, y_multi = [], []
        ind_ls = np.random.choice(cls_num[j],size=(2,cls_num[i]), replace=True)
        #ind_ls = np.random.randint(cls_num[j],size=(2,cls_num[i]))
        while it<2:     
            #ind_ls = np.random.randint(cls_num[j],size=cls_num[i])                 
            batch_x2 = X2[j][ind_ls[it,:],:,:,:]            
            batch_y2 = Y2[j][ind_ls[it]]            
            batch_y2 = np_utils.to_categorical(batch_y2, 10)
            batch_x, batch_y = multi_batch(batch_x1, batch_y1, batch_x2, batch_y2)
            if len(x_multi)==0: 
                x_multi=batch_x
                y_multi=batch_y
            else:
                x_multi=np.concatenate((x_multi,batch_x), axis=0)
                y_multi=np.concatenate((y_multi,batch_y), axis=0)
            it+=1
            batch_x2, batch_y2, batch_x, batch_y = None, None, None, None           
            del batch_x2
            del batch_y2
            del batch_x
            del batch_y
        print("finished inner inner 2 round")
        ind_ls = None
        del ind_ls

        ind_test = np.random.choice(cls_num[i]*2,size=2000, replace=False)
        #ind_test = np.random.randint(cls_num[i]*2,size=2000)
        if len(x_train)==0: 
            x_test = x_multi[ind_test,:,:,:]
            y_test = y_multi[ind_test,:]
            x_multi = x_multi.tolist()
            y_multi = y_multi.tolist()
            for ele in sorted(ind_test, reverse = True):
                x_multi.pop(ele)
                y_multi.pop(ele)
            x_train = x_multi
            y_train = y_multi
            print("finished split test and train set at ",j,"round")
        else:
            x_test = np.concatenate((x_test,x_multi[ind_test,:,:,:]), axis=0)
            y_test = np.concatenate((y_test,y_multi[ind_test,:]), axis=0)
            x_multi = x_multi.tolist()
            y_multi = y_multi.tolist()
            for ele in sorted(ind_test, reverse = True):
                x_multi.pop(ele)
                y_multi.pop(ele)
            x_train = np.concatenate((x_train,x_multi), axis=0)
            y_train = np.concatenate((y_train,y_multi), axis=0)
            print("finished split test and train set at ",j,"round")
        ind_test, x_multi, y_multi  = None, None, None
        del ind_test
        del x_multi
        del y_multi 
        print("inner",j,"chunk is finished.")
    print("outer",i,"chunk is finished.")
    batch_x1, batch_y1 = None, None
    del batch_x1
    del batch_y1

# save dataset as text file
#x_train = list(x_train)
#x_test  = list(x_test)

with open('./multi_mnist.txt', 'w') as f:
    f.write(str(x_train)+'\n')
    f.write(str(y_train)+'\n')
    f.write(str(x_test) +'\n')
    f.write(str(y_test))

print("Job Finished.")
print("x_train shape: ", np.shape(x_train))
print("y_train shape: ", np.shape(y_train))
print("x_test  shape: ", np.shape(x_test))
print("y_test  shape: ", np.shape(y_test))
x_train, y_train, x_test, y_test = None, None, None, None
del x_train
del y_train
del x_test
del y_test

##########################################

