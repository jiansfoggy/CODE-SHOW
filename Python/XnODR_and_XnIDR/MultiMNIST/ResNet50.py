from __future__ import absolute_import, division, print_function
import os,re,sys,json,time,keras
import PIL.Image, PIL.ImageFont, PIL.ImageDraw
import numpy as np

import tensorflow as tf
from keras.utils import np_utils
from tensorflow.keras.applications.resnet50 import ResNet50
from matplotlib import pyplot as plt


#K.set_image_data_format('channels_last')  # for capsule net
#K.clear_session()
print("Tensorflow version " + tf.__version__)

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
    batch_x  = (batch_x1 + batch_x2)

    #batch_y = np.clip(batch_y1+batch_y2, 0, 1)
    batch_y  = (batch_y1 + batch_y2)*2-1

    return batch_x, batch_y, batch_x1, batch_x2

BATCH_SIZE = 100
classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

#Matplotlib config
plt.rc('image', cmap='gray')
plt.rc('grid', linewidth=0)
plt.rc('xtick', top=False, bottom=False, labelsize='large')
plt.rc('ytick', left=False, right=False, labelsize='large')
plt.rc('axes', facecolor='F8F8F8', titlesize="large", edgecolor='white')
plt.rc('text', color='a8151a')
plt.rc('figure', facecolor='F0F0F0')# Matplotlib fonts
MATPLOTLIB_FONT_DIR = os.path.join(os.path.dirname(plt.__file__), "mpl-data/fonts/ttf")

# utility to display a row of digits with their predictions
def display_images(digits, predictions, labels, title):

    n = 10

    indexes = np.random.choice(len(predictions), size=n)
    n_digits = digits[indexes]
    n_predictions = predictions[indexes]
    n_predictions = n_predictions.reshape((n,))
    n_labels = labels[indexes]
 
    fig = plt.figure(figsize=(20, 4))
    plt.title(title)
    plt.yticks([])
    plt.xticks([])

    for i in range(10):
        ax = fig.add_subplot(1, 10, i+1)
        class_index = n_predictions[i]
      
        plt.xlabel(classes[class_index])
        plt.xticks([])
        plt.yticks([])
        plt.imshow(n_digits[i])

# utility to display training and validation curves
def plot_metrics(metric_name, title, ylim=5):
    plt.title(title)
    plt.ylim(0,ylim)
    plt.plot(history.history[metric_name],color='blue',label=metric_name)
    plt.plot(history.history['val_' + metric_name],color='green',label='val_' + metric_name)

def stats_graph(graph):
    flops = tf.compat.v1.profiler.profile(graph, options=tf.compat.v1.profiler.ProfileOptionBuilder.float_operation())
    params = tf.compat.v1.profiler.profile(graph,
                                           options=tf.compat.v1.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    print('FLOPs: {};    Trainable params: {}'.format(flops.total_float_ops,
                                                      params.total_parameters))
def preprocess_image_input(input_images):
    #print("input iamges",np.shape(input_images))
    #input_images = np.stack((input_images)*3, axis=-1)
    input_images = np.einsum('kzij->zijk', input_images)
    output_ims = tf.keras.applications.resnet50.preprocess_input(input_images)
    #print("input iamges",np.shape(output_ims))
    return output_ims

def feature_extractor(inputs):
    feature_extractor = tf.keras.applications.resnet.ResNet50(input_shape=(224, 224, 3),
                                               include_top=False,
                                               weights='imagenet')(inputs)
    return feature_extractor

def classifier(inputs):
    #print(np.shape(inputs))
    x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1024, activation="relu")(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.Dense(10, activation="softmax", name="classification")(x)
    return x

def final_model(inputs):
    resize = tf.keras.layers.UpSampling2D(size=(3,3))(inputs)
    resnet_feature_extractor = feature_extractor(resize)
    classification_output = classifier(resnet_feature_extractor)

    return classification_output

def define_compile_model():
    inputs = tf.keras.layers.Input(shape=(36,36,3))
    classification_output = final_model(inputs) 
    model = tf.keras.Model(inputs=inputs, outputs = classification_output)
 
    model.compile(optimizer='SGD', 
                  loss='squared_hinge',
                  #loss='sparse_categorical_crossentropy',
                  #metrics = [tf.keras.metrics.TopKCategoricalAccuracy(k=2)])
                  metrics = ['accuracy'])
  
    return model

with tf.Graph().as_default() as graph:
    (x_train, y_train) , (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') 
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') 
    #display_images(training_images, training_labels, training_labels, "Training Data" )
    #display_images(validation_images, validation_labels, validation_labels, "Validation Data" )
    
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

    repeat_time=5
    totl_smpl=repeat_time*70000
    trn_smpl =(repeat_time-1)*60000
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

    uni_ind = []
    for ui in range(totl_smpl):
        if Y1[ui] != Y2[ui]:
            uni_ind.append(ui)

    X1 = X1[uni_ind,:,:,:] 
    X2 = X2[uni_ind,:,:,:] 
    Y1 = Y1[uni_ind]
    Y2 = Y2[uni_ind]

    print("Unique Image Shape is:", np.shape(X1), np.shape(X2))
    print("Unique Label Shape is:", np.shape(Y1), np.shape(Y2))

    batch_y1 = np_utils.to_categorical(Y1, 10)
    batch_y2 = np_utils.to_categorical(Y2, 10)
    y1_train = np.vstack([Y1[:trn_smpl], Y2[:trn_smpl]]).T
    y1_test  = np.vstack([Y1[trn_smpl:280000], Y2[trn_smpl:280000]]).T
    Y1, Y2   = None, None
    del Y1
    del Y2
    batch_x, batch_y, X1, X2 = multi_batch(X1, batch_y1, X2, batch_y2)
    batch_y1, batch_y2 = None, None
    del batch_y1
    del batch_y2

    x_train  = batch_x[:trn_smpl,:,:,:]
    x_train  = [x_train[:,:,:,0],x_train[:,:,:,0],x_train[:,:,:,0]]
    x_test   = batch_x[trn_smpl:280000,:,:,:]
    x_test   = [x_test[:,:,:,0],x_test[:,:,:,0],x_test[:,:,:,0]]
    batch_x  = None
    del batch_x
    x1_train = X1[:trn_smpl,:,:,:]
    x1_train = [x1_train[:,:,:,0],x1_train[:,:,:,0],x1_train[:,:,:,0]]
    x2_train = X2[:trn_smpl,:,:,:]
    x2_train = [x2_train[:,:,:,0],x2_train[:,:,:,0],x2_train[:,:,:,0]]
    x1_test  = X1[trn_smpl:280000,:,:,:]
    x1_test  = [x1_test[:,:,:,0],x1_test[:,:,:,0],x1_test[:,:,:,0]]
    x2_test  = X2[trn_smpl:280000,:,:,:]
    x2_test  = [x2_test[:,:,:,0],x2_test[:,:,:,0],x2_test[:,:,:,0]]
    X1, X2   = None, None
    del X1
    del X2

    y_train  = batch_y[:trn_smpl,:]
    y_test   = batch_y[trn_smpl:280000,:]
    #y_train  = y1_train
    #y_test   = y1_test
    batch_y  = None
    del batch_y

    print("Job Finished.")
    print("x_train shape: ", np.shape(x_train))
    print("y_train shape: ", np.shape(y_train))
    print("x_test  shape: ", np.shape(x_test))
    print("y_test  shape: ", np.shape(y_test))

    x_train = preprocess_image_input(x_train)
    x_test = preprocess_image_input(x_test)
    x1_train = preprocess_image_input(x1_train)
    x1_test = preprocess_image_input(x1_test)
    x2_train = preprocess_image_input(x2_train)
    x2_test = preprocess_image_input(x2_test)

    model = define_compile_model()

    model.summary()

    EPOCHS = 15
    t_start = time.time()
    history = model.fit(x_train, y_train, batch_size=100, epochs=EPOCHS,
                        verbose=1, validation_data=(x_test, y_test))
    #history = model.fit(x_train, y_train, epochs=EPOCHS, validation_data = (x_test, y_test), batch_size=100)
    del x_train
    del y_train
    del x1_train
    del x2_train
    del y1_train
    del x1_test
    del x2_test
    loss, accuracy = model.evaluate(x_test, y_test, batch_size=100)
    y_pred = model.predict([x_test], batch_size=100)  
    top1 = np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1))/y_test.shape[0]
    top2 = tf.reduce_mean(tf.keras.metrics.top_k_categorical_accuracy(y_test, y_pred, k=2))
    top2 = top2.eval(session=tf.compat.v1.Session())   
    top5 = tf.reduce_mean(tf.keras.metrics.top_k_categorical_accuracy(y_test, y_pred, k=5))
    top5 = top5.eval(session=tf.compat.v1.Session())
    t_end   = time.time()
    del x_test
    del y_test
    del y1_test

    stats_graph(graph)
print("loss: ", loss)
print("Top1 Acc: ", top1)
print("Top2 Acc: ", top2)
print("Top5 Acc: ", top5)
print("Running Time is: ", t_end-t_start)
print(x_test[0:10])
'''
1
FLOPs: 53175701;    Trainable params: 26162698
loss:  0.8645660677552223
Top1 Acc:  0.5166
Top2 Acc:  0.965925
Top5 Acc:  0.996725
Running Time is:  8298.654069662094

2
FLOPs: 53175701;    Trainable params: 26162698
loss:  0.8748045924305916
Top1 Acc:  0.59845
Top2 Acc:  0.971525
Top5 Acc:  0.99765
Running Time is:  10351.9458193779

3
FLOPs: 53175701;    Trainable params: 26162698
loss:  0.8645924822986126
Top1 Acc:  0.50375
Top2 Acc:  0.964925
Top5 Acc:  0.996925
Running Time is:  10870.05610203743

4
FLOPs: 53175701;    Trainable params: 26162698
loss:  0.8644055292010308
Top1 Acc:  0.4992
Top2 Acc:  0.964975
Top5 Acc:  0.996825
Running Time is:  7958.604628801346

5
FLOPs: 53175701;    Trainable params: 26162698
loss:  0.8649192282557487
Top1 Acc:  0.5011
Top2 Acc:  0.964875
Top5 Acc:  0.99635
Running Time is:  8362.15176653862

'''
