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

def preprocess_image_input(input_images):
    input_images = input_images.astype('float32')
    input_images = np.stack((input_images,)*3, axis=-1)
    output_ims = tf.keras.applications.resnet50.preprocess_input(input_images)
    return output_ims

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

def RES_Primary(x, filters, n_channels, dim_vector): 
    #renet block where dimension doesnot change.
    #The skip connection is just simple identity conncection
    #we will have 3 blocks and then input will be added

    x_skip = x # this will be used for addition with the residual block 
    f1, f2 = filters
  
    #first block 
    x = Conv2D(f1, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=regularizers.l2(0.001))(x)
    x = BatchNormalization()(x)
    #x = BatchNormalization(epsilon=1e-6, momentum=0.9, axis=-1)(x)
    #x = layers.advanced_activations.ReLU()(x)
    x = layers.Activation('relu')(x)
    #x = ReLU()(x)

    #second block # bottleneck (but size kept same with padding)
    x = Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=regularizers.l2(0.001))(x)
    #x = BatchNormalization(epsilon=1e-6, momentum=0.9, axis=-1)(x)
    x = BatchNormalization()(x)
    #x = layers.advanced_activations.ReLU()(x)
    x = layers.Activation('relu')(x)
    #x = ReLU()(x)

    # third block activation used after adding the input
    x = Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=regularizers.l2(0.001))(x)
    #x = BatchNormalization(epsilon=1e-6, momentum=0.9, axis=-1)(x)
    x = BatchNormalization()(x)
    # x = Activation(activations.relu)(x)

    # add the input 
    x = Add()([x, x_skip])
    #x = Conv2D(f1, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=regularizers.l2(0.001))(x)
    #x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = AvgPool2D (pool_size = 4, strides = 1, data_format='channels_last')(x)
    x = layers.Reshape(target_shape=[n_channels, dim_vector])(x)
    x = layers.Lambda(squash)(x)
    x = BatchNormalization()(x)
    #x = layers.advanced_activations.ReLU()(x)
    x = layers.Activation('relu')(x)
    #x = ReLU()(x)
    return x
"""
def RES_Primary(x, n_channels, dim_vector):   

    #x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = AvgPool2D (pool_size = 7, strides = 1, data_format='channels_last')(x)
    x = layers.Reshape(target_shape=[n_channels, dim_vector])(x)
    x = layers.Lambda(squash)(x) 
    #x = BatchNormalization(epsilon=1e-6, momentum=0.9, axis=-1)(x)
    x = BatchNormalization()(x)
    x = layers.advanced_activations.ReLU()(x)
    #x = layers.Activation('relu')(x)
    return x
"""
def feature_extractor(inputs):
    
    mbln = tf.keras.applications.resnet.ResNet50(input_shape=(224, 224, 3),
                                               include_top=False,
                                               weights='imagenet')
    for layer in mbln.layers[-10:]:
        layer.trainable=False
    feature_extractor = mbln(inputs)
    """
    feature_extractor = tf.keras.applications.resnet.ResNet50(input_shape=(224, 224, 3),
                                               include_top=False,
                                               weights='imagenet')(inputs)
    """
    return feature_extractor

def classifier(inputs):
    #x = RES_Primary(inputs, filters=2048, dim_vector=8, n_channels=256)
    x = RES_Primary(inputs, filters=(512, 2048), dim_vector=8, n_channels=256)
    #x = RES_Primary(inputs, dim_vector=8, n_channels=256)
    XDR4 = XDR1_LPLayer(num_capsule=10, dim_vector=16, num_routing=3, name='XDR4')(x)
    Bn4 = layers.BatchNormalization(name='Bn4')(XDR4)
    Act4 = layers.Activation('relu',name='Act4')(Bn4)
    #Act4 = layers.advanced_activations.ReLU(name='Act4')(Bn4)
    out_cpxn = Length(name='out_cpxn')(Act4)
    return out_cpxn

def final_model(inputs):
    resize = tf.keras.layers.UpSampling2D(size=(3,3))(inputs)

    resnet_feature_extractor = feature_extractor(resize)
    classification_output = classifier(resnet_feature_extractor)

    return classification_output

def define_compile_model():
    inputs = tf.keras.layers.Input(shape=(36,36,3))
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
    #batch_y  = (batch_y1 + batch_y2)

    return batch_x, batch_y, batch_x1, batch_x2

def test(model, data):
    
    x_test, y_test = data
    #, x_recon [x_test, y_test]
    y_pred = model.predict([x_test], batch_size=100)
    top1 = np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1))/y_test.shape[0]
    top2 = tf.reduce_mean(tf.keras.metrics.top_k_categorical_accuracy(y_test, y_pred, k=2))
    top2 = top2.eval(session=tf.compat.v1.Session())   
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
    return a, b, x_test, top1, top2, top5

with tf.Graph().as_default() as graph:
    # download MNIST dataset from keras
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # convert data type to float 32
    x_train = x_train.reshape(-1, 28, 28).astype('float32') / 255.
    x_test  = x_test.reshape(-1, 28, 28).astype('float32') / 255.
    x_train = np.stack((x_train,)*3, axis=-1)
    x_test  = np.stack((x_test,)*3, axis=-1)
    n_class = 10

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
    x_test   = batch_x[trn_smpl:280000,:,:,:]
    batch_x  = None
    del batch_x
    x1_train = X1[:trn_smpl,:,:,:]
    x2_train = X2[:trn_smpl,:,:,:]
    x1_test  = X1[trn_smpl:280000,:,:,:]
    x2_test  = X2[trn_smpl:280000,:,:,:]
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

    model = define_compile_model()

    model.summary()
    log = callbacks.CSVLogger('./Log_Res/ORES_ODR_log.csv')
    checkpoint = callbacks.ModelCheckpoint('./Log_Res/ORES_ODR_weight.h5',
                                           save_best_only=True, mode='max',
                                           save_weights_only=True, verbose=1)
    lr_schdl = CyclicLR(mode='triangular2')

    EPOCHS = 20
    t_start = time.time()
    history = model.fit(x_train, y_train, epochs=EPOCHS, verbose=1,
        validation_data = (x_test, y_test), batch_size=100,
        callbacks=[log, checkpoint, lr_schdl])
    t_start = time.time()
    #loss, accuracy = model.evaluate(x_test, y_test, batch_size=80)
    a,b,x_test,top1,top2,top5 = test(model=model, data=(x_test, y_test))  
    t_end   = time.time()
    stats_graph(graph)
print('-'*50)
print("Top 1 Test Acc: ", top1)
print("Top 2 Test Acc: ", top2)
print("Top 5 Test Acc: ", top5)
print('-'*50)
print("Running Time is: ", t_end-t_start)
print('-'*50)

'''
Total params: 28,387,296
Trainable params: 23,862,320
Non-trainable params: 4,524,976

FLOPs: 98857278;    Trainable params: 28327984
--------------------------------------------------
Top 1 Test Acc:  0.649275
Top 2 Test Acc:  0.992425
Top 5 Test Acc:  1.0
--------------------------------------------------
Running Time is:  47.076435565948486
--------------------------------------------------

'''
