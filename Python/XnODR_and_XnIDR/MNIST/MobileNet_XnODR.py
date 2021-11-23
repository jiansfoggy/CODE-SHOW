import os, re, time, json, sys
import PIL.Image, PIL.ImageFont, PIL.ImageDraw
import numpy as np

import tensorflow as tf
import keras.backend as K
from matplotlib import pyplot as plt
from keras.datasets import mnist
from keras.optimizers import Adam
from keras.models import Sequential
from keras.utils import np_utils
# from keras.optimizers import SGD, Adam, RMSprop
from tensorflow.python.framework import ops
from keras.utils.vis_utils import plot_model
from keras import callbacks, initializers, layers, models, constraints, optimizers
from keras.layers import Dense, DepthwiseConv2D, Dropout, Activation, BatchNormalization, MaxPooling2D
from keras.layers import Flatten, InputSpec, Layer, Conv2D, AvgPool2D, ReLU
from keras.preprocessing.image import ImageDataGenerator
from clr_callback import *

K.set_image_data_format('channels_last')  # for capsule net
K.clear_session()
print("Tensorflow version " + tf.__version__)

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

def round_through(x):
    rounded = K.round(x)
    return x + K.stop_gradient(rounded - x)

def _hard_sigmoid(x):
    x = (0.5 * x) + 0.5
    return K.clip(x, 0, 1)

def binary_sigmoid(x):
    return round_through(_hard_sigmoid(x))

def binary_tanh(x):
    return 2 * round_through(_hard_sigmoid(x)) - 1

def binarize(W, H=1):
    # [-H, H] -> -H or H
    Wb = H * binary_tanh(W / H)
    return Wb

def _mean_abs(x, axis=None, keepdims=False):
    return K.stop_gradient(K.mean(K.abs(x), axis=axis, keepdims=keepdims))

def xnorize(W, H=1., axis=None, keepdims=False):
    Wb = binarize(W, H)
    Wa = _mean_abs(W, axis, keepdims)   
    return Wa, Wb

class Length(layers.Layer):
    """
    Compute the length of vectors. This is used to compute a Tensor that has the same shape with y_true in margin_loss
    inputs: shape=[dim_1, ..., dim_{n-1}, dim_n]
    output: shape=[dim_1, ..., dim_{n-1}]
    """
    def call(self, inputs, **kwargs):
        return K.sqrt(K.sum(K.square(inputs), -1))

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]

def squash(vectors, axis=-1):

    s_squared_norm = K.sum(K.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / K.sqrt(s_squared_norm)
    return scale * vectors

def margin_loss(y_true, y_pred):
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred))+0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))
    return K.mean(K.sum(L, 1))

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

class XDR1_LPLayer(layers.Layer):

    def __init__(self, num_capsule, dim_vector, num_routing=3,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 **kwargs):
        super(XDR1_LPLayer, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_vector = dim_vector
        self.num_routing = num_routing
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

    def build(self, input_shape):
        assert len(input_shape) >= 3, "The input Tensor should have shape=[None, input_num_capsule, input_dim_vector]"
        self.input_num_capsule = input_shape[1]
        self.input_dim_vector = input_shape[2]

        shape=[self.input_num_capsule, self.num_capsule, self.input_dim_vector, self.dim_vector]
        #print("within build capsule layer input_shape and shape",input_shape, shape)
        
        self.W = self.add_weight(shape=[self.input_num_capsule, self.num_capsule, self.input_dim_vector, self.dim_vector],
                                 initializer=self.kernel_initializer,
                                 name='W')
        # Coupling coefficient. The redundant dimensions are just to facilitate subsequent matrix calculation.
        self.bias = self.add_weight(shape=[1, self.input_num_capsule, self.num_capsule, 1, 1],
                                    initializer=self.bias_initializer,
                                    name='bias',
                                    trainable=False)
        self.built = True

    def call(self, inputs, training=None):
        inputs_expand = K.expand_dims(K.expand_dims(inputs, 2), 2)
        inputs_tiled = K.tile(inputs_expand, [1, 1, self.num_capsule, 1, 1])
        
        x_a, x_b = xnorize(inputs_tiled, 1., axis=4, keepdims=True) # (nb_sample, 1)
        w_a, w_b = xnorize(self.W, 1., axis=2, keepdims=True) # (1, units)

        inputs_hat = tf.scan(lambda ac,x: tf.matmul(x, w_b), 
                         elems=(x_b),
                         initializer=tf.zeros([self.input_num_capsule, self.num_capsule, 1, self.dim_vector]))

        inputs_hat *= x_a*w_a

        assert self.num_routing > 0, 'The num_routing should be > 0.'
        for i in range(self.num_routing):
            c = tf.nn.softmax(self.bias, axis=2)
            outputs = squash(K.sum(c * inputs_hat, 1, keepdims=True))

            # last iteration needs not compute bias which will not be passed to the graph any more anyway.
            if i != self.num_routing - 1:
                self.bias = self.bias + K.sum(inputs_hat * outputs, -1, keepdims=True)
        return K.reshape(outputs, [-1, self.num_capsule, self.dim_vector])

    def compute_output_shape(self, input_shape):
        return tuple([None, self.num_capsule, self.dim_vector])

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
    input_images = np.stack((input_images,)*3, axis=-1)
    output_ims = tf.keras.applications.mobilenet_v2.preprocess_input(input_images)
    return output_ims

def MBL_Primary(x, dim_vector, n_channels, strides):
    
    x = DepthwiseConv2D(kernel_size = 3, strides = strides, padding = 'same')(x)
    #x = BatchNormalization(epsilon=1e-6, momentum=0.9, axis=-1)(x)
    x = BatchNormalization()(x)
    #x = ReLU()(x)

    x = Conv2D(filters = dim_vector*n_channels, kernel_size = 1, strides = 1)(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)    
    #x = AvgPool2D (pool_size = 7, strides = 1, data_format='channels_last')(x)
    #x = tf.nn.max_pool(x, ksize=[1, 7, 7, 1], strides=[1, 1, 1, 1], padding='VALID')
    #x = tf.keras.layers.Dropout(0.5)(x)
    x = layers.Reshape(target_shape=[n_channels, dim_vector])(x)
    x = layers.Lambda(squash)(x)
    #x = BatchNormalization(epsilon=1e-6, momentum=0.9, axis=-1)(x)
    x = BatchNormalization()(x)
    #x = ReLU()(x)
    x = layers.advanced_activations.ReLU()(x)
    return x

def feature_extractor(inputs):
    mbln = tf.keras.applications.mobilenet_v2.MobileNetV2(input_shape=(224, 224, 3),
                                               include_top=False, weights='imagenet')
    for layer in mbln.layers[-5:]:
        layer.trainable=False
    feature_extractor = mbln(inputs)
    return feature_extractor

def classifier(inputs):
    x = MBL_Primary(inputs, dim_vector=8, n_channels=128, strides = 1)
    x = XDR1_LPLayer(num_capsule=10, dim_vector=16, num_routing=3)(x)
    x = tf.keras.layers.Dropout(.3)(x)
    #Bn4 = layers.BatchNormalization(epsilon=epsilon, momentum=momentum, axis=channel_axis, name='Bn4')(XDR4)
    Bn4 = layers.BatchNormalization(name='Bn4')(x)
    #x = layers.Activation('relu')(Bn4)
    x = layers.advanced_activations.ReLU()(Bn4)   
    x = Length(name='out_cpxn')(x)
    return x

def final_model(inputs):
    inputs = tf.keras.layers.ZeroPadding2D(padding=(2, 2))(inputs)
    resize = tf.keras.layers.UpSampling2D(size=(7,7))(inputs)

    resnet_feature_extractor = feature_extractor(resize)
    classification_output = classifier(resnet_feature_extractor)

    '''
    mbln = tf.keras.applications.mobilenet_v2.MobileNetV2(input_shape=(224, 224, 3),
                                               include_top=False, weights='imagenet')
    i = 0
    for layer in mbln.layers[:149]:
        if i == 0:
            x = layer(resize)
            i+=1
        else: x = layer(x)
    classification_output = classifier(x)
    '''
    return classification_output

def define_compile_model():
    inputs = tf.keras.layers.Input(shape=(28,28,3))
  
    classification_output = final_model(inputs) 
    model = tf.keras.Model(inputs=inputs, outputs = classification_output)
    adamm = optimizers.Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, amsgrad=False)
    
    model.compile(optimizer=adamm, 
                  #optimizer="SGD", 
                  #loss=[margin_loss], 
                  loss=["squared_hinge"], 
                  #loss='sparse_categorical_crossentropy',
                  metrics = ['accuracy'])
  
    return model

def test(model, data):
    
    x_test, y_test = data
    #, x_recon [x_test, y_test]
    y_pred = model.predict([x_test], batch_size=150)
    top1 = np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1))/y_test.shape[0]
    top5 = tf.reduce_mean(tf.keras.metrics.top_k_categorical_accuracy(y_test, y_pred, k=5))
    top5 = top5.eval(session=tf.compat.v1.Session())
    """
    print('-'*50)
    print('Top1 Test Acc:', np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1))/y_test.shape[0])
    print('Top5 Test Acc:', top5)
    """
    a=np.argmax(y_test, 1)
    b=np.argmax(y_pred, 1)
    #, x_recon
    return a, b, x_test, top1, top5

with tf.Graph().as_default() as graph:
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = preprocess_image_input(x_train)
    x_test = preprocess_image_input(x_test)
    
    y_train = np_utils.to_categorical(y_train, 10)  # -1 or 1 for hinge loss
    y_test = np_utils.to_categorical(y_test, 10)
    #y_train = np_utils.to_categorical(y_train, 10)*2-1  # -1 or 1 for hinge loss
    #y_test = np_utils.to_categorical(y_test, 10)*2-1
    
    log = callbacks.CSVLogger('./Log_Mbl/OMBODR_log.csv')
    checkpoint = callbacks.ModelCheckpoint('./Log_Mbl/OMBODR_weight.h5',
                                           save_best_only=True, mode='max',
                                           save_weights_only=True, verbose=1)
    #lr_schdl = callbacks.LearningRateScheduler(schedule=lambda e: lr_start * lr_decay ** e)
    lr_schdl = CyclicLR(mode='triangular2')
    
    model = define_compile_model()

    model.summary()

    EPOCHS = 1
    before_T = time.time()
    history = model.fit(x_train, y_train, epochs=EPOCHS, 
        validation_data = (x_test, y_test), batch_size=150,
        callbacks=[log, lr_schdl, checkpoint])
    #loss, accuracy = model.evaluate(valid_X, validation_labels, batch_size=150)
    before_T = time.time()
    a,b,x_test,top1,top5 = test(model=model, data=(x_test, y_test))
    after_T = time.time()
    stats_graph(graph)
print('-'*50)
print('Top1 Test Acc:', top1)
print('Top5 Test Acc:', top5)
print('-'*50)
print("The total training time is: ", (after_T-before_T), " seconds.")
print('-'*50)

"""
1

FLOPs: 12672973;    Trainable params: 3714864
--------------------------------------------------
Top1 Test Acc: 0.839
Top5 Test Acc: 0.9996
--------------------------------------------------
The total training time is:  13.299367427825928  seconds.
--------------------------------------------------

FLOPs: 12672974;    Trainable params: 3714864 Trainable params: 2,994,864
--------------------------------------------------
Top1 Test Acc: 0.9968
Top5 Test Acc: 1.0
--------------------------------------------------
The total training time is:  14.730234384536743  seconds.
--------------------------------------------------

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

ResNet-50
1
--------------------------------------------------

"""
