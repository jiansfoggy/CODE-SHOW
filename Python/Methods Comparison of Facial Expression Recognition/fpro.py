'''
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from sklearn.metrics import confusion_matrix
import itertools
get_ipython().run_line_magic('matplotlib', 'inline')
'''
################
# Load Package #
################
#path = "/Users/sunjian/Downloads/libsvm-3.23/python"
#sys.path.append(path)
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import pandas as pd
import tensorflow as tf
from numpy.linalg import *
#matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from keras.optimizers import Adam
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers.convolutional import *
import os, sys, cv2, random, sklearn, keras
from sklearn.preprocessing import StandardScaler
from keras.metrics import categorical_crossentropy
from sklearn.model_selection import train_test_split
from keras import callbacks, layers, optimizers, models 
from keras.layers.normalization import BatchNormalization
from sklearn.metrics import classification_report, confusion_matrix
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Activation, Dropout, Flatten 


'''
# mv -T Manually_Annotated_file_list_cropped_\(not_expanded\) path_label
# mv Manually_Annotated_Images_cropped_\(not_expanded\) imageset
# cp validation_set_v2_cropped_list.csv /home/user1/
# dataset/path_label/valid_list.csv
# cp training_set_v2_cropped_list.csv /home/user1/
# dataset/path_label/train_list.csv

# cp -avr /home/user1/affectNet/Manually_Annotated_Images_cropped_\
# (not_expanded\)/ /home/user1/dataset/
# subDirectory+filePath,Face_x,Face_y,Face_width,
# Face_height,facialLandMarks,Expression,valence,arousal
'''
##################
# Load Data File #
##################
train_file = pd.read_csv('/home/user1/dataset/path_label/train_list.csv')
valid_file = pd.read_csv('/home/user1/dataset/path_label/valid_list.csv')

train_file = train_file.rename(columns = {"subDirectory+filePath": "filepath",
                                          "Expression":"label"})
valid_file = valid_file.rename(columns = {"subDirectory+filePath": "filepath",
                                          "Expression":"label"})

ttl_data1 = train_file[train_file.label < 8][['filepath','label']]
ttl_data2 = valid_file[valid_file.label < 8][['filepath','label']]
jg1 = int(input("We have %d training images. How many do you want to train: " %(len(ttl_data1))))
train_path = list(ttl_data1.filepath)[0:jg1]
train_label = train_file.label[0:jg1]
jg2 = int(input("We have %d validation images. How many do you want to valid: " %(len(ttl_data2))))
valid_path = list(valid_file.filepath)[0:jg2]
valid_label = valid_file.label[0:jg2]

y_train, y_valid = list(train_label), list(valid_label)

def load_img(path_list):
    num = len(path_list)
    wh = int(input("The resized wideth or height: "))
    emptyset = np.zeros((len(path_list), wh*wh))
    step1 = "/home/user1/dataset/imageset"
    for i in range(len(path_list)):
        step2 = [step1, path_list[i]]
        pwd='/'.join(step2)
        getimg = cv2.imread(pwd)
        gray_img = cv2.cvtColor(getimg, cv2.COLOR_BGR2GRAY)
        dim = (wh, wh)
        gra_img = cv2.resize(gray_img, dim, interpolation = cv2.INTER_NEAREST)
        flat_img = np.reshape(gra_img,(1,wh*wh))
        emptyset[i] = np.float32(flat_img)/255.

    return emptyset, num

x_train, train_num = load_img(train_path)
x_valid, valid_num = load_img(valid_path)
print(np.shape(x_train),np.shape(x_valid))

x_test, y_test = x_valid, y_valid
label_dict = {0: 'Neutral', 1: 'Happy', 2: 'Sad', 3: 'Surprise', 4: 'Fear',
              5: 'Disgust', 6: 'Anger', 7: 'Contempt'}

############## 
# Class Part #
############## 
# Build PCA Class
class SJPCA(object):
    def __init__(self):
        pass

    def train(self, X):
        self.x_train = X

    def compute_mean_covar_eigen(self):
        # get average image and get mean image by summing each row
        tr_mean = np.mean(self.x_train, axis=0)
        tr_mean = np.reshape(tr_mean,(1,np.shape(tr_mean)[0]))

        # subtract the mean
        xtr_m = self.x_train - tr_mean
        # calculate covariance matrix
        tr_cov = np.dot(xtr_m.T,xtr_m)
        # get eigenvalue and eigenvector
        tr_val, tr_vec = eig(tr_cov)
        return xtr_m, tr_cov, tr_val, tr_vec

    def get_comp_K(self,tr_val, threshold):
        cum_lambda = np.cumsum(tr_val)
        total_lamda = cum_lambda[-1]

        # get the principal component number that we want to keep
        for keep_dim in range(len(tr_val)):
            rate = cum_lambda[keep_dim]/total_lamda
            if rate >= threshold:
                return keep_dim
                break
            else: continue

    def deduct_img(self, xtr_m, tr_vec, keep_dim):
        x_proj= np.dot(xtr_m, tr_vec.T[:,0:keep_dim])
        return x_proj

# Build KNN Class
class SJKNN(object):
    def __init__(self):
        pass

    def train(self, X, Y):
    # the nearest neighbor classifier simply remembers all the training data
        self.X_train = X
        self.Y_train = Y

    def compute_distances_no_loops(self, X_test):
        num_test = np.shape(X_test)[0]
        num_train = np.shape(self.X_train)[0]
        dists = np.zeros((num_test, num_train))
        dists = np.sqrt(self.getNormMatrix(X_test, num_train).T +
                        self.getNormMatrix(self.X_train, num_test) -
                        2 * np.dot(X_test, self.X_train.T))
        pass
        return(dists)
    
    def getNormMatrix(self, x, lines_num):
        return(np.ones((lines_num, 1)) * np.sum(np.square(x), axis = 1))

    def predict_labels(self, dists, k):
        num_test = np.shape(dists)[0]
        Y_pred = np.zeros(num_test)
        for i in range(num_test):
            closest_y = []
            kids = np.argsort(dists[i])
            #print(kids)
            closest_y = np.array(self.Y_train)[kids[:k]]
            count = 0
            label = 0
            for j in closest_y:
                tmp = 0
                for kk in closest_y:
                    tmp += (kk == j)
                if tmp > count:
                    count = tmp
                    label = j
            Y_pred[i] = label
        return Y_pred

    def predict(self, X_test, k):
        num_test = X_test.shape[0]
        # lets make sure that the output type matches the input type
        #ypred = np.zeros(num_test, dtype = self.Y_train.dtype)
        ypred = np.zeros(num_test)
        dists = self.compute_distances_no_loops(X_test)
        return self.predict_labels(dists, k=k)

# Build LR Class
class SJLogis_Regre(object):
    def __init__(self):
        pass

    def train(self, X, Y):
        self.x_train = X
        self.y_train = Y

    def split_category1(self, category_name):
        yy_train=[]
        for i in range(len(y_train)):
            if (self.y_train[i]==category_name):
                yy_train.append(1)
            else: yy_train.append(0)
        return yy_train

    def sigmoid(self, a):
        return 1 / (1 + np.exp(-a))

    def log_likelihood1(self, ytrain_c, weight):

        # add intercept
        intercept = np.ones((np.shape(self.x_train)[0], 1))
        xtrain_c = np.hstack((intercept, self.x_train))

        weight = np.reshape(weight,(np.shape(xtrain_c)[1], 1))
        a = np.dot(xtrain_c, weight)

        ll = np.sum( np.multiply(ytrain_c,a.T) - np.log(1+np.exp(a.T)) )
        return ll

    def gradient_descent1(self, ytrain_c, learning_rate, iteration_time):

        # add intercept
        intercept = np.ones((np.shape(self.x_train)[0], 1))
        xtrain_c = np.hstack((intercept, self.x_train))

        # initial weight
        weight = np.zeros((1,np.shape(xtrain_c)[1]))
        ytrain_c = np.reshape(ytrain_c,(1, np.shape(ytrain_c)[0]))

        # do iteration
        for i in range(iteration_time):
            a = np.dot(weight, xtrain_c.T)
            pred = self.sigmoid(a)

            diff = ytrain_c - pred

            gradient = np.dot(diff, xtrain_c)
            weight = weight + learning_rate * gradient

            # Print the cost
            if (i % 10000 == 0):
                cost = -self.log_likelihood1(ytrain_c, weight)

                print ("the cost in %d step is %3f" %(i,cost))

        return weight

    def get_pcx(self, xtest_c, weight_c):

        add_intercept = np.hstack((1, xtest_c))

        p_c = np.dot(weight_c,add_intercept)
        result_c = self.sigmoid(p_c)
        return result_c

    def predict_c(self, total_result):
        value = np.where(total_result == np.max(total_result))
        return value[0][0]


choose_method = input("Choose one, 1:KNN, 2:LR, 3:ANN, 4:CNN -- ")
if (int(choose_method)==1 or int(choose_method)== 2):
#######################
# Dimension Reduction #
#######################
    # stack train and valid as a big one for dimension deduction
    big_X=np.vstack((x_train,x_test))
    SJ = SJPCA()
    SJ.train(big_X)
    xtr_m, tr_cov, tr_val, tr_vec = SJ.compute_mean_covar_eigen()
    threshold_pca = input("The percentage that you want to keep: ")
    keep_dim = SJ.get_comp_K(tr_val, float(threshold_pca))
    new_big_X = SJ.deduct_img(xtr_m, tr_vec, keep_dim)
    print("The kept dimension is",keep_dim)

    # resplit the dataset and normalize them with min-max normalization
    x_train = new_big_X[0:train_num,:]
    x_test = new_big_X[train_num:train_num+valid_num,:]
    tr_min = np.min(x_train,axis=1)
    tr_cha = np.max(x_train,axis=1)-np.min(x_train,axis=1)
    te_min = np.min(x_test,axis=1)
    te_cha = np.max(x_test,axis=1)-np.min(x_test,axis=1)
    for i in range(train_num):
        x_train[i]=(x_train[i]-tr_min[i])/tr_cha[i]
    for j in range(valid_num):
        x_test[j]=(x_test[j]-te_min[j])/te_cha[j]

###########
# Try KNN #
###########
    if (int(choose_method)==1):
        # select best k
        K = []
        n = int(input("Enter number of K: "))
        for lst in range(0, n):
            ele = int(input())
            K.append(ele) # adding the element     

        SJ = SJKNN()
        SJ.train(x_train, y_train)
        num_test = len(y_test)
        Acc_lst = []
        for k_value in K:
            Y_test_pred=SJ.predict(x_test, k=k_value)
            num_correct = np.sum(Y_test_pred == y_test)
            print('Got %d / %d correct' % (num_correct, num_test))
            k_acc=np.mean(y_test == Y_test_pred)
            Acc_lst.append(k_acc)
            print('k = %s, Accuracy = %4f' % (k_value, k_acc))

        bestk = K[np.where(Acc_lst==np.max(Acc_lst))[0][0]]

        Y_test_pred=SJ.predict(x_test, k=bestk)
        num_correct = np.sum(Y_test_pred == y_test)
        print('Got %d / %d correct' % (num_correct, num_test))
        print('k = %s, Accuracy = %f' % (5, np.mean(y_test == Y_test_pred)))
        print(confusion_matrix(y_test, Y_test_pred))
        print(classification_report(y_test, Y_test_pred,
              target_names=list(label_dict.values()),digits=3))
        plt.figure(figsize=(8,8))
        cnf_matrix = confusion_matrix(y_test, Y_test_pred)
        classes = list(label_dict.values())
        plt.imshow(cnf_matrix, interpolation='nearest')
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        _ = plt.xticks(tick_marks, classes, rotation=90)
        _ = plt.yticks(tick_marks, classes)
 
##########
# Try LR #
##########   
    else:
        # split the train as 10 categories
        JS = SJLogis_Regre()
        JS.train(x_train,y_train)
        y0_train = JS.split_category1(0)
        y1_train = JS.split_category1(1)
        y2_train = JS.split_category1(2)
        y3_train = JS.split_category1(3)
        y4_train = JS.split_category1(4)
        y5_train = JS.split_category1(5)
        y6_train = JS.split_category1(6)
        y7_train = JS.split_category1(7)
        
        import time
        # calculate weight seperately
        learning_rate = float(input("Please write down learning rate: "))
        iteration_time = float(input("Please write the iteration times: "))
        tic = time.time()
        w0 = JS.gradient_descent1(y0_train, learning_rate, iteration_time)
        w1 = JS.gradient_descent1(y1_train, learning_rate, iteration_time)
        w2 = JS.gradient_descent1(y2_train, learning_rate, iteration_time)
        w3 = JS.gradient_descent1(y3_train, learning_rate, iteration_time)
        w4 = JS.gradient_descent1(y4_train, learning_rate, iteration_time)
        w5 = JS.gradient_descent1(y5_train, learning_rate, iteration_time)
        w6 = JS.gradient_descent1(y6_train, learning_rate, iteration_time)
        w7 = JS.gradient_descent1(y7_train, learning_rate, iteration_time)
        toc = time.time()
        print('Iteration took %f seconds' %(toc - tic))

        # calculate probability for each category 
        y_pred = []
        for i in range(np.shape(x_test)[0]):
            pred_0 = JS.get_pcx(x_test[i], w0)
            pred_1 = JS.get_pcx(x_test[i], w1)
            pred_2 = JS.get_pcx(x_test[i], w2)
            pred_3 = JS.get_pcx(x_test[i], w3)
            pred_4 = JS.get_pcx(x_test[i], w4)
            pred_5 = JS.get_pcx(x_test[i], w5)
            pred_6 = JS.get_pcx(x_test[i], w6)
            pred_7 = JS.get_pcx(x_test[i], w7)

            pred = [pred_0[0],pred_1[0],pred_2[0],pred_3[0],pred_4[0],
                    pred_5[0],pred_6[0],pred_7[0],pred_8[0],pred_9[0]]
            value = JS.predict_c(pred)
            y_pred.append(value)
        # calculate accuracy
        num_test = len(y_test)
        num_correct = np.sum(y_pred == y_test)
        print('Got %d / %d correct' % (num_correct, num_test))
        print('Accuracy = %f' % (np.mean(y_test == y_pred)))
        print(confusion_matrix(y_test, y_pred)) 
        print(classification_report(y_test, y_pred,
              target_names=list(label_dict.values()),digits=3))
        plt.figure(figsize=(8,8))
        cnf_matrix = confusion_matrix(y_test, y_pred)
        classes = list(label_dict.values())
        plt.imshow(cnf_matrix, interpolation='nearest')
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        _ = plt.xticks(tick_marks, classes, rotation=90)
        _ = plt.yticks(tick_marks, classes)

###########
# Try CNN #
###########
elif (int(choose_method)==4):
    y_train_labels = to_categorical(y_train)
    y_test_labels = to_categorical(y_test)
    # Convert the images into 3 channels 
    X_train=np.dstack([x_train] * 3) 
    X_test=np.dstack([x_test] * 3) 
    print("The shape of new train: ",np.shape(X_train), ",The shape of new test: ",np.shape(X_test))
    # Reshape images as per the tensor format required by tensorflow 
    wh = int(input("input the size again: "))
    X_train = X_train.reshape(-1,wh,wh,3)
    X_test = X_test.reshape (-1,wh,wh,3)
    print("The shape of new train: ",np.shape(X_train), ",The shape of new test: ",np.shape(X_test))

    # Define the parameters for instanitaing model
    IMG_WIDTH = int(input("The image width is: "))
    IMG_HEIGHT = int(input("The image height is: "))
    IMG_DEPTH = int(input("The image layers is: "))
    BATCH_SIZE = int(input("The number for each batch is: "))

    model = Sequential()

    # 1st Convolutional Layer
    model.add(Conv2D(filters=96, input_shape=(IMG_WIDTH, IMG_HEIGHT, IMG_DEPTH), kernel_size=(5,5),\
     strides=(4,4), padding='valid'))
    model.add(Activation('relu'))
    # Pooling 
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    # Batch Normalisation before passing it to the next layer
    model.add(BatchNormalization())

    # 2nd Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))
    # Pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    # Batch Normalisation
    model.add(BatchNormalization())

    # 3rd Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))
    # Batch Normalisation
    model.add(BatchNormalization())

    # 4th Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))
    # Batch Normalisation
    model.add(BatchNormalization())

    # 5th Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))
    # Pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    # Batch Normalisation
    model.add(BatchNormalization())

    # Passing it to a dense layer
    model.add(Flatten())
    # 1st Dense Layer
    model.add(Dense(4096, input_shape=(IMG_WIDTH, IMG_HEIGHT, IMG_DEPTH)))
    model.add(Activation('relu'))
    # Add Dropout to prevent overfitting
    model.add(Dropout(0.4))
    # Batch Normalisation
    model.add(BatchNormalization())

    # 2nd Dense Layer
    model.add(Dense(4096))
    model.add(Activation('relu'))
    # Add Dropout
    model.add(Dropout(0.4))
    # Batch Normalisation
    model.add(BatchNormalization())

    # 3rd Dense Layer
    model.add(Dense(1000))
    model.add(Activation('relu'))
    # Add Dropout
    model.add(Dropout(0.4))
    # Batch Normalisation
    model.add(BatchNormalization())

    # Output Layer
    model.add(Dense(8))
    model.add(Activation('softmax'))

    NB_EPOCHS = int(input("The epoach number is: "))
    
    from keras import models
    from keras.models import Model
    from keras import optimizers
    from keras import callbacks
    # Compile the model.
    model.compile(Adam(lr=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    # Incorporating reduced learning and early stopping for callback
    reduce_learning = callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.2, patience=2,
        verbose=1, mode='auto', epsilon=0.0001,
        cooldown=2, min_lr=0)

    eary_stopping = callbacks.EarlyStopping(
        monitor='val_loss', min_delta=0.0003,
        patience=8, verbose=1,
        mode='auto')

    callbacks = [reduce_learning, eary_stopping]
    # Train the the model
    mt=model.fit(X_train, y_train_labels,
        epochs=NB_EPOCHS,
        validation_data=(X_test, y_test_labels),
        callbacks=callbacks)
    # Evaluate accuracy
    test_loss, test_acc = model.evaluate(X_test, y_test_labels)
    print('Test Accuracy:%4f, Test Loss:%4f.' %(test_acc,test_loss))

    # Make Prediction
    pred_result = model.predict(X_test)
    y_pred=[]
    for i in range(np.shape(y_test)[0]):
        num = np.where(pred_result[i]==max(pred_result[i]))
        y_pred.append(num[0][0])
    y_pred = np.transpose(y_pred)
    # calculate accuracy
    num_test = len(y_test)
    num_correct = np.sum(y_pred == y_test)
    print('Got %d / %d correct' % (num_correct, num_test))
    print('Accuracy = %f' % (np.mean(y_test == y_pred)))
    print(confusion_matrix(y_test, y_pred)) 
    print(classification_report(y_test, y_pred,
          target_names=list(label_dict.values()),digits=3))
else:
    print("Wrong Input. Please try 1,2,3,4")
'''

    train_features = model.predict(np.array(X_train), batch_size=BATCH_SIZE, verbose=1) 
    test_features = model.predict(np.array(X_test), batch_size=BATCH_SIZE, verbose=1)
    
    np.savez("train_features", train_features, y_train_label)
    np.savez("test_features", test_features, y_test_labels)
    # Current shape of features
    print(train_features.shape, "\n",  test_features.shape, "\n")

    # Flatten extracted features
    train_features_flat = np.reshape(train_features, (jg1, 1*1*512))
    test_features_flat = np.reshape(test_features, (jg2, 1*1*512))
'''



