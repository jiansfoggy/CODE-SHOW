import os, re, time, json
import PIL.Image, PIL.ImageFont, PIL.ImageDraw
import numpy as np

import tensorflow as tf
from keras import callbacks
from tensorflow.keras.applications.resnet50 import ResNet50
from matplotlib import pyplot as plt

#K.set_image_data_format('channels_last')  # for capsule net
#K.clear_session()
print("Tensorflow version " + tf.__version__)

BATCH_SIZE = 150 
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

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
    input_images = input_images.astype('float32')
    output_ims = tf.keras.applications.resnet50.preprocess_input(input_images)
    return output_ims

def feature_extractor(inputs):

    feature_extractor = tf.keras.applications.resnet.ResNet50(input_shape=(224, 224, 3),
                                               include_top=False)(inputs)
    #feature_extractor = tf.keras.applications.resnet.ResNet50(input_shape=(224, 224, 3),
    #                                           include_top=False,
    #                                           weights='imagenet')(inputs)
    return feature_extractor

def classifier(inputs):
    #x = tf.keras.layers.AvgPool2D (pool_size = 7, strides = 1, data_format='channels_last')(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1024, activation="relu")(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.Dense(10, activation="softmax", name="classification")(x)
    return x

def final_model(inputs):
    resize = tf.keras.layers.UpSampling2D(size=(7,7))(inputs)
    #resize = tf.keras.layers.UpSampling2D(size=(2,2))(inputs)

    resnet_feature_extractor = feature_extractor(resize)
    classification_output = classifier(resnet_feature_extractor)

    return classification_output

def define_compile_model():
    inputs = tf.keras.layers.Input(shape=(32,32,3))
  
    classification_output = final_model(inputs) 
    model = tf.keras.Model(inputs=inputs, outputs = classification_output)
 
    model.compile(optimizer='SGD', 
                  loss='sparse_categorical_crossentropy',
                  metrics = ['accuracy'])
  
    return model

with tf.Graph().as_default() as graph:
    (training_images, training_labels) , (validation_images, validation_labels) = tf.keras.datasets.cifar10.load_data()

    display_images(training_images, training_labels, training_labels, "Training Data" )
    display_images(validation_images, validation_labels, validation_labels, "Validation Data" )

    train_X = preprocess_image_input(training_images)
    valid_X = preprocess_image_input(validation_images)

    model = define_compile_model()

    model.summary()
    log = callbacks.CSVLogger('./Log_Res/RES_log.csv')
    checkpoint = callbacks.ModelCheckpoint('./Log_Res/RES_weight.h5',
                                           save_best_only=True, verbose=1,
                                           monitor='val_accuracy', mode='max',
                                           save_weights_only=True)
    

    EPOCHS = 30
    
    history = model.fit(train_X, training_labels, epochs=EPOCHS, 
                        validation_data = (valid_X, validation_labels), 
                        batch_size=64, callbacks=[log, checkpoint])
    print("The list of record that can be plotted out:", history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='lower right')
    plt.savefig('./Log_Res/RES_accuracy.png')
    plt.clf()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.savefig('./Log_Res/RES_loss.png')
    plt.clf()

    t_start = time.time()
    loss, accuracy = model.evaluate(valid_X, validation_labels, batch_size=64)
    t_end   = time.time()
    stats_graph(graph)
print("loss: ", loss)
print("accuracy: ", accuracy)
print("Running Time is: ", t_end-t_start)

'''
without pretrain weight
FLOPs: 52375700;    Trainable params: 26162698
loss:  0.20541727979928254
accuracy:  0.9558
Running Time is:  10068.749220132828

1
FLOPs: 52375700;    Trainable params: 26162698
loss:  0.19435020726919175
accuracy:  0.9539

2
FLOPs: 52375700;    Trainable params: 26162698
loss:  0.1797980574414134
accuracy:  0.9572

3
FLOPs: 52375700;    Trainable params: 26162698
loss:  0.19790267332196235
accuracy:  0.9539

4
FLOPs: 52375700;    Trainable params: 26162698
loss:  0.1878259834855795
accuracy:  0.9556

5
FLOPs: 52375700;    Trainable params: 26162698
loss:  0.19686801243722438
accuracy:  0.9528

6
FLOPs: 52375700;    Trainable params: 26162698
loss:  0.17677949094623327
accuracy:  0.9586

# %tensorflow_version 1.x
from __future__ import absolute_import, division, print_function
import os, sys, time, keras
import numpy as np
import tensorflow as tf
import keras.backend as K

tf.compat.v1.disable_eager_execution()
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.optimizers import Adam
from keras.models import Sequential
from keras.utils import np_utils, to_categorical
from tensorflow.python.framework import ops
from keras.utils.vis_utils import plot_model
from keras import callbacks, initializers, layers, models, constraints, optimizers
from keras.layers import Dense, Dropout, Activation, BatchNormalization, MaxPooling2D
from keras.layers import Flatten, InputSpec, Layer, Conv2D
from keras.preprocessing.image import ImageDataGenerator

K.set_image_data_format('channels_last')  # for capsule net
K.clear_session()

def stats_graph(graph):
    flops = tf.compat.v1.profiler.profile(graph, options=tf.compat.v1.profiler.ProfileOptionBuilder.float_operation())
    params = tf.compat.v1.profiler.profile(graph,
                                           options=tf.compat.v1.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    print('FLOPs: {};    Trainable params: {}'.format(flops.total_float_ops,
                                                      params.total_parameters))

class Clip(constraints.Constraint):
    def __init__(self, min_value, max_value=None):
        self.min_value = min_value
        self.max_value = max_value
        if not self.max_value:
            self.max_value = -self.min_value
        if self.min_value > self.max_value:
            self.min_value, self.max_value = self.max_value, self.min_value

    def __call__(self, p):
        return K.clip(p, self.min_value, self.max_value)

    def get_config(self):
        return {"min_value": self.min_value,
                "max_value": self.max_value}

# download MNIST dataset from keras
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
# convert data type to float 32
x_train = x_train.reshape(-1, 32, 32, 3).astype('float32') / 255.
x_test = x_test.reshape(-1, 32, 32, 3).astype('float32') / 255.
n_class = 10
y_train = np_utils.to_categorical(y_train, n_class)  # -1 or 1 for hinge loss
y_test = np_utils.to_categorical(y_test, n_class)
print("Modified Input Image Shape:", np.shape(x_train))
print("Modified Input Label Shape:", np.shape(y_train))

def train(model, data, NB_EPOCHS, batch_size):
    # unpacking the data
    (x_train, y_train), (x_test, y_test) = data
    # learning rate schedule
    lr_start = 1e-3
    lr_end = 1e-6
    lr_decay = (lr_end / lr_start) ** (1. / NB_EPOCHS)

    # callbacks
    log = callbacks.CSVLogger('./Without_Recon/Caps_log.csv')
    checkpoint = callbacks.ModelCheckpoint('./Without_Recon/Caps_weight.h5',
                                           save_best_only=True,
                                           save_weights_only=True, verbose=1)
    lr_schdl = callbacks.LearningRateScheduler(schedule=lambda e: lr_start * lr_decay ** e)
    # lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: 0.0001 * np.exp(-epoch / 10.))
    reduce_learning = callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.2, patience=2,
        verbose=1, mode='auto', epsilon=0.0001,
        cooldown=2, min_lr=0)
    eary_stopping = callbacks.EarlyStopping(
        monitor='val_loss', min_delta=0.002,
        patience=7, verbose=1,
        mode='auto')

    # compile the model
    adamm = optimizers.Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(optimizer=adamm,
                  # loss=comprehensive_loss, loss_weights=[1.0],
                  loss=['categorical_crossentropy'], loss_weights=[1.0],
                  # loss=[margin_loss,'mse'],
                  # loss_weights=[1.0, 0.0005],
                  metrics={'predictions': 'accuracy'})

    # create data generator
    # datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
    # prepare iterator
    # datagen.fit(x_train)
    # it_train = datagen.flow(x_train, y_train, batch_size=batch_size)
    # fit model
    # steps = int(x_train.shape[0] / batch_size)
    # model.load_weights('trained_model.h5')
    # '''
    # history = model.fit_generator(it_train, steps_per_epoch=steps, epochs=NB_EPOCHS,
    #                              validation_data=(x_test, y_test), verbose=1,
    #                              callbacks=[log, lr_schdl])
'''
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=NB_EPOCHS,
                        verbose=1, validation_data=[x_test, y_test],
                        callbacks=[log, lr_schdl])

    model.save_weights('./Without_Recon/Caps_weight.h5')
    print('Trained model saved to \'./Without_Recon/Caps_weight.h5\'')

    # dict_keys(['val_loss', 'val_acc', 'loss', 'acc', 'lr'])
    print("The list of record that can be plotted out:", history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    # plt.plot(history.history['out_cpxn_accuracy'])
    # plt.plot(history.history['val_out_cpxn_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='lower right')
    # plt.show()
    plt.savefig('./Without_Recon/Caps_accuracy.png')
    plt.clf()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    # plt.plot(history.history['out_cpxn_loss'])
    # plt.plot(history.history['val_out_cpxn_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    # plt.show()
    plt.savefig('./Without_Recon/Caps_loss.png')
    plt.clf()
    return model

def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(np.sqrt(num))
    height = int(np.ceil(float(num) / width))
    shape = generated_images.shape[1:3]
    image = np.zeros((height * shape[0], width * shape[1], 3),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index / width)
        j = index % width
        image[i * shape[0]:(i + 1) * shape[0], j * shape[1]:(j + 1) * shape[1], :] = \
            img[:, :, :]
    return image

def test(model, data):
    x_test, y_test = data
    # , x_recon [x_test, y_test]
    y_pred = model.predict([x_test], batch_size=150)
    top1 = np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1)) / y_test.shape[0]
    top5 = tf.reduce_mean(tf.keras.metrics.top_k_categorical_accuracy(y_test, y_pred, k=5))
    top5 = top5.eval(session=tf.compat.v1.Session())

    a = np.argmax(y_test, 1)
    b = np.argmax(y_pred, 1)
    # , x_recon
    return a, b, x_test, top1, top5

from sklearn.metrics import classification_report, confusion_matrix
# def plott(a,b,x_test,x_recon):
def plott(a, b, x_test):
    label_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer',
                  5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}
    print(confusion_matrix(a, b))
    print(classification_report(a, b,
                                target_names=list(label_dict.values()), digits=3))

with tf.Graph().as_default() as graph:

    x = layers.Input(shape=[32, 32, 3])

    model = tf.keras.applications.resnet50.ResNet50(
        include_top=True, weights=None, input_tensor=x,
        input_shape=[32, 32, 3], pooling='avg', classes=10,classifier_activation='softmax')

    model.summary()

    try:
        plot_model(model, to_file='./Without_Recon/Caps_structure.png', show_shapes=True, show_layer_names=True)
    except Exception as e:
        print('No fancy plot {}'.format(e))

    before_T = time.time()
    train(model=model, data=((x_train, y_train), (x_test, y_test)),
          NB_EPOCHS=60, batch_size=150)
    after_T = time.time()
    a, b, x_test, top1, top5 = test(model=model, data=(x_test, y_test))
    plott(a, b, x_test)
    stats_graph(graph)
print('-' * 50)
print('Top1 Test Acc:', top1)
print('Top5 Test Acc:', top5)
print('-' * 50)
print("The total training time is: ", (after_T - before_T), " seconds.")
print('-' * 50)
'''
"""


=======================
Without Reconstruct
=======================

top1    = [86.40,86.42,86.51,86.15,86.09]
top5    = [99.15,99.05,99.12,99.13,99.01]
time_ls = [6819.390365600586,7252.462275266647,6946.476444482803,7026.700607538223,7025.691478967667]

Standard Deviation
Top1 86.31400000000001 0.1637803407005859
Top5 99.092 0.053065996645686335
Time 7014.144234371185 141.13432549923922

Total params: 26,215,818
Trainable params: 26,162,698
Non-trainable params: 53,120

without pretrained weight
FLOPs: 52375699;    Trainable params: 26162698
loss:  0.6773234145343303
accuracy:  0.8781
Running Time is:  1750.1087670326233

ResNet-50
1
FLOPs: 47157201;    Trainable params: 23555082
--------------------------------------------------
Top1 Test Acc: 0.7017
Top5 Test Acc: 0.967
--------------------------------------------------
The total training time is:  2068.968818426132  seconds.
--------------------------------------------------

2
FLOPs: 47157201;    Trainable params: 23555082
--------------------------------------------------
Top1 Test Acc: 0.7107
Top5 Test Acc: 0.9701
--------------------------------------------------
The total training time is:  2045.4242782592773  seconds.
--------------------------------------------------

3
FLOPs: 47157201;    Trainable params: 23555082
--------------------------------------------------
Top1 Test Acc: 0.715
Top5 Test Acc: 0.969
--------------------------------------------------
The total training time is:  1762.2917103767395  seconds.
--------------------------------------------------
"""
