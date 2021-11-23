import os, re, time, json
import PIL.Image, PIL.ImageFont, PIL.ImageDraw
import numpy as np

import tensorflow as tf
import keras.backend as K
from tensorflow.keras.applications.resnet50 import ResNet50
from matplotlib import pyplot as plt
from keras.utils import np_utils
from keras import callbacks, initializers, regularizers, layers, models, constraints, optimizers
from keras.layers import Add, Dense, DepthwiseConv2D, Dropout, Activation, BatchNormalization, MaxPooling2D
from keras.layers import Flatten, InputSpec, Layer, Conv2D, AvgPool2D, ReLU
from keras.preprocessing.image import ImageDataGenerator
from clr_callback import *

K.set_image_data_format('channels_last')  # for capsule net
K.clear_session()
print("Tensorflow version " + tf.__version__)

BATCH_SIZE = 80
classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

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

def preprocess_image_input(input_images):
    input_images = input_images.astype('float32')
    input_images = np.stack((input_images,)*3, axis=-1)
    output_ims = tf.keras.applications.resnet50.preprocess_input(input_images)
    return output_ims

class XDR2_LPLayer(layers.Layer):

    def __init__(self, num_capsule, dim_vector, num_routing=3,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 **kwargs):
        super(XDR2_LPLayer, self).__init__(**kwargs)
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

        self.W = self.add_weight(shape=[self.input_num_capsule, self.num_capsule, self.input_dim_vector, self.dim_vector],
                                 initializer=self.kernel_initializer,
                                 name='W')
        self.bias = self.add_weight(shape=[1, self.input_num_capsule, self.num_capsule, 1, 1],
                                    initializer=self.bias_initializer,
                                    name='bias',
                                    trainable=False)
        self.built = True

    def call(self, inputs, training=None):
        inputs_expand = K.expand_dims(K.expand_dims(inputs, 2), 2)
        inputs_tiled = K.tile(inputs_expand, [1, 1, self.num_capsule, 1, 1])
        
        inputs_hat = tf.scan(lambda ac,x: tf.matmul(x, self.W), 
                         elems=(inputs_tiled),
                         initializer=tf.zeros([self.input_num_capsule, self.num_capsule, 1, self.dim_vector]))

        assert self.num_routing > 0, 'The num_routing should be > 0.'
        for i in range(self.num_routing):
            c = tf.nn.softmax(self.bias, axis=2)
            outputs = squash(K.sum(c * inputs_hat, 1, keepdims=True))

            if i != self.num_routing - 1:
                x_a, x_b = xnorize(inputs_hat, 1., axis=4, keepdims=True) # (nb_sample, 1)
                w_a, w_b = xnorize(outputs, 1., axis=4, keepdims=True) # (1, units)

                self.bias = self.bias + K.sum(x_b * w_b * x_a * w_a, -1, keepdims=True)
        return K.reshape(outputs, [-1, self.num_capsule, self.dim_vector])

    def compute_output_shape(self, input_shape):
        return tuple([None, self.num_capsule, self.dim_vector])

def RES_Primary(x, filters, n_channels, dim_vector): 
    x_skip = x # this will be used for addition with the residual block 
    f1, f2 = filters
  
    #first block 
    x = Conv2D(f1, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=regularizers.l2(0.001))(x)
    x = BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    #second block # bottleneck (but size kept same with padding)
    x = Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=regularizers.l2(0.001))(x)
    x = BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # third block activation used after adding the input
    x = Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=regularizers.l2(0.001))(x)
    x = BatchNormalization()(x)

    # add the input 
    x = Add()([x, x_skip])
    x = AvgPool2D (pool_size = 7, strides = 1, data_format='channels_last')(x)
    x = layers.Reshape(target_shape=[n_channels, dim_vector])(x)
    x = layers.Lambda(squash)(x)
    x = BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x

def feature_extractor(inputs):
    
    mbln = tf.keras.applications.resnet.ResNet50(input_shape=(224, 224, 3),
                                               include_top=False,
                                               weights='imagenet')
    for layer in mbln.layers[-10:]:
        layer.trainable=False
    feature_extractor = mbln(inputs)
    return feature_extractor

def classifier(inputs):
    x = RES_Primary(inputs, filters=(512, 2048), dim_vector=8, n_channels=256)
    XDR4 = XDR2_LPLayer(num_capsule=10, dim_vector=16, num_routing=3, name='XDR4')(x)
    Bn4 = layers.BatchNormalization(name='Bn4')(XDR4)
    Act4 = layers.Activation('relu',name='Act4')(Bn4)
    out_cpxn = Length(name='out_cpxn')(Act4)
    return out_cpxn

def final_model(inputs):
    inputs = tf.keras.layers.ZeroPadding2D(padding=(2, 2))(inputs)
    resize = tf.keras.layers.UpSampling2D(size=(7,7))(inputs)

    resnet_feature_extractor = feature_extractor(resize)
    classification_output = classifier(resnet_feature_extractor)

    return classification_output

def define_compile_model():
    inputs = tf.keras.layers.Input(shape=(28,28,3))
    classification_output = final_model(inputs) 
    model = tf.keras.Model(inputs=inputs, outputs = classification_output)
    adamm = optimizers.Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(#optimizer='SGD', 
                  optimizer=adamm,
                  #loss='sparse_categorical_crossentropy',
                  #loss=[margin_loss],
                  loss=['squared_hinge'],
                  metrics = ['accuracy'])
  
    return model


(training_images, training_labels) , (validation_images, validation_labels) = tf.keras.datasets.mnist.load_data()

train_X = preprocess_image_input(training_images)
valid_X = preprocess_image_input(validation_images)
#training_labels = np_utils.to_categorical(training_labels, 10) # -1 or 1 for hinge loss
#validation_labels = np_utils.to_categorical(validation_labels, 10)
training_labels = np_utils.to_categorical(training_labels, 10)*2-1  # -1 or 1 for hinge loss
validation_labels = np_utils.to_categorical(validation_labels, 10)*2-1

model = define_compile_model()

model.summary()
log = callbacks.CSVLogger('./Log_Res/ORES_IDR_log.csv')
checkpoint = callbacks.ModelCheckpoint('./Log_Res/ORES_IDR_weight.h5',
                                       save_best_only=True, mode='max',
                                       save_weights_only=True, verbose=1)
lr_schdl = CyclicLR(mode='triangular2')

EPOCHS = 20
t_start = time.time()
history = model.fit(train_X, training_labels, epochs=EPOCHS, verbose=1,
    validation_data = (valid_X, validation_labels), batch_size=80,
    callbacks=[log, checkpoint, lr_schdl])
t_start = time.time()
loss, accuracy = model.evaluate(valid_X, validation_labels, batch_size=80)
t_end   = time.time()

print("loss: ", loss)
print("accuracy: ", accuracy)
print("Running Time is: ", t_end-t_start)


