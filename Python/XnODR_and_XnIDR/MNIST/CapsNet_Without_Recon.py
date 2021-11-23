#%tensorflow_version 1.x
from __future__ import absolute_import, division, print_function
import os,sys,time,keras
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
#from tensorflow.python.eager import profiler
import keras.backend as K
tf.compat.v1.disable_eager_execution()
#from tensorflow import keras
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.optimizers import Adam
from keras.models import Sequential
from keras.utils import np_utils, to_categorical
# from keras.optimizers import SGD, Adam, RMSprop
from tensorflow.python.framework import ops
from keras.utils.vis_utils import plot_model
from keras import callbacks, initializers, layers, models, constraints, optimizers
from keras.layers import Dense, Dropout, Activation, BatchNormalization, MaxPooling2D
from keras.layers import Flatten, InputSpec, Layer, Conv2D
from keras.preprocessing.image import ImageDataGenerator
#K.set_image_data_format('channels_first') # for xnor net
K.set_image_data_format('channels_last') # for capsule net
K.clear_session()

def stats_graph(graph):
    flops = tf.compat.v1.profiler.profile(graph, options=tf.compat.v1.profiler.ProfileOptionBuilder.float_operation())
    params = tf.compat.v1.profiler.profile(graph, options=tf.compat.v1.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    print('FLOPs: {};    Trainable params: {}'.format(flops.total_float_ops,
          params.total_parameters))

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

class Mask(layers.Layer):

    def call(self, inputs, **kwargs):
        # use true label to select target capsule, shape=[batch_size, num_capsule]
        if type(inputs) is list:  # true label is provided with shape = [batch_size, n_classes], i.e. one-hot code.
            assert len(inputs) == 2
            inputs, mask = inputs
            print("inputs, mask",inputs, mask)
        else:  # if no true label, mask by the max length of vectors of capsules
            x = inputs
            # Enlarge the range of values in x to make max(new_x)=1 and others < 0
            x = (x - K.max(x, 1, True)) / K.epsilon() + 1
            mask = K.clip(x, 0, 1)  # the max value in x clipped to 1 and other to 0
            print("mask",mask)

        # masked inputs, shape = [batch_size, dim_vector]
        # inputs_masked = tf.matmul(inputs, mask)
        inputs_masked = K.batch_dot(mask, inputs, [1, 1])
        return inputs_masked

    def compute_output_shape(self, input_shape):
        if type(input_shape[0]) is tuple:  # true label provided
            return tuple([None, input_shape[0][-1]])
        else:
            return tuple([None, input_shape[-1]])

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

def PrimaryCap(inputs, dim_vector, n_channels, kernel_size, strides, padding):
    input_shape = np.shape(inputs)
    print("Within PrimaryCap input shape: ", input_shape)
    output_size = int((input_shape[1] - kernel_size + 2*0)//2) +1
        
    output  = layers.Conv2D(filters=dim_vector*n_channels, kernel_size=kernel_size, strides=strides, padding=padding)(inputs)
    print("Within PrimaryCap output shape: ", np.shape(output))
    maxpl = tf.nn.max_pool(output, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='maxpl')
    outputs = layers.Reshape(target_shape=[output_size*output_size*n_channels, dim_vector])(maxpl)
        
    return layers.Lambda(squash)(outputs)

class CapsLayer(layers.Layer):

    def __init__(self, num_capsule, dim_vector, num_routing=3,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 **kwargs):
        super(CapsLayer, self).__init__(**kwargs)
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
        
        #inputs_hat = tf.scan(lambda ac,x: tf.matmul(x, self.W), 
        #                     elems=inputs_tiled,
        #                     initializer=tf.zeros([self.input_num_capsule, self.num_capsule, 1, self.dim_vector]))
       
        inputs_hat = tf.map_fn(lambda x: tf.matmul(x, self.W), 
                             elems=inputs_tiled)
        self.bias = tf.zeros(shape=[self.input_num_capsule, self.num_capsule, 1, self.dim_vector]) 

        # Routing algorithm V2. Use iteration. V2 and V1 both work without much difference on performance
        assert self.num_routing > 0, 'The num_routing should be > 0.'
        for i in range(self.num_routing):
            c = tf.nn.softmax(self.bias, axis=2)
            #c = tf.nn.softmax(self.bias, dim=2)  # dim=2 is the num_capsule dimension
            # outputs.shape=[None, 1, num_capsule, 1, dim_vector]
            outputs = squash(K.sum(c * inputs_hat, 1, keepdims=True))
            # last iteration needs not compute bias which will not be passed to the graph any more anyway.
            if i != self.num_routing - 1:
                self.bias = self.bias + K.sum(inputs_hat * outputs, -1, keepdims=True)
        return K.reshape(outputs, [-1, self.num_capsule, self.dim_vector])

    def compute_output_shape(self, input_shape):
        return tuple([None, self.num_capsule, self.dim_vector])


# download MNIST dataset from keras
(x_train, y_train), (x_test, y_test1) = tf.keras.datasets.mnist.load_data()
# convert data type to float 32
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.
n_class = 10
#y_train = np_utils.to_categorical(y_train, n_class) * 2 - 1 # -1 or 1 for hinge loss
#y_test = np_utils.to_categorical(y_test, n_class) *2 - 1
y_train = np_utils.to_categorical(y_train, n_class)  # -1 or 1 for hinge loss
y_test = np_utils.to_categorical(y_test1, n_class)
print("Modified Input Image Shape:",np.shape(x_train))
print("Modified Input Label Shape:",np.shape(y_train))

def CpXnNet (input_shape1,n_class,epsilon,momentum,use_bias,num_routing):
    # Set channel axis and Input shape
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    x = layers.Input(shape=input_shape1)   
    # Layer 1
    print("-------------- First Layer : Convolutional Layer ---------------")
    conv1 = layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='valid', name='conv1')(x)
    bn1 = layers.BatchNormalization(epsilon=epsilon, momentum=momentum,axis=channel_axis, name='bn1')(conv1)
    act1 = layers.Activation('relu', name='act1')(bn1)
    print("the output shape of activation layer is:",np.shape(act1),"     |")
    print("--------------------- End of First Layer -----------------------\n")
    
    # Layer 2
    print("-------------- Second Layer : Convolutional Layer --------------")
    input_shape2 = np.shape(act1)
    conv2 = layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='valid', name='conv2')(act1)
    maxpool2 = layers.MaxPooling2D(pool_size=(2, 2),strides=(2,2),name='maxpool2')(conv2)
    bn2 = layers.BatchNormalization(epsilon=epsilon, momentum=momentum,axis=channel_axis, name='bn2')(maxpool2)
    act2 = layers.Activation('relu', name='act2')(bn2)
    print("the output shape of maxpooling layer is:",np.shape(act2),"     |")
    print("--------------------- End of Second Layer ----------------------\n")
    
    # Layer 3
    print("-------------- Third Layer : Convolutional Layer ---------------")
    input_shape3 = np.shape(act2)
    conv3 = layers.Conv2D(filters=256, kernel_size=3, strides=1, padding='valid', name='conv3')(act2)
    bn3 = layers.BatchNormalization(epsilon=epsilon, momentum=momentum,axis=channel_axis, name='bn3')(conv3)
    act3 = layers.Activation('relu', name='act3')(bn3)
    print("the output shape of convolution layer is:",np.shape(act3),"        |")
    print("--------------------- End of Third Layer -----------------------\n")

    # Layer 4
    print("-------------- Fourth Layer : Fully Connected Layer ------------")
    input_shape4 = np.shape(act3)
    conv4 = PrimaryCap(act3, dim_vector=8, n_channels=32, kernel_size=3, strides=1, padding='valid')
    bn4 = layers.BatchNormalization(epsilon=epsilon, momentum=momentum, axis=channel_axis, name='bn4')(conv4)
    act4 = layers.Activation('relu',name='act4')(bn4)
    print("the output shape of activation layer is:",np.shape(act4),"            |")
    print("--------------------- End of Fourth Layer ----------------------\n")

    # Layer 5
    print("-------------- Fifth Layer : Fully Connected Layer -------------")
    input_shape5 = np.shape(act4)
    fc5 = CapsLayer(num_capsule=n_class, dim_vector=16, num_routing=num_routing, name='fc5')(act4)
    bn5 = layers.BatchNormalization(epsilon=epsilon, momentum=momentum, axis=channel_axis, name='bn5')(fc5)
    act5 = layers.Activation('relu',name='act5')(bn5)
    #print(np.array(act5))
    print("the output shape of fully connected layer is:",np.shape(act5),"         |")
    print("--------------------- End of Fifth Layer -----------------------\n")
    
    # Layer 6
    print("-------------- Sixth Layer : Auxiliary Layer -----------------")
    out_cpxn = Length(name='out_cpxn')(act5)
    print("the output shape of flatten layer is:",np.shape(out_cpxn),"               |")
    print("--------------------- End of Auxiliary Layer -------------------\n")
    #'''
    # Specify Input and Output.
    print("-------------- Output Check : Four Elements --------------------")
    print("X and Y Input:", np.shape(x),"                         |")
    print("X and Y Output:", np.shape(out_cpxn),"                        |")
    print("--------------------- End of Output Check ----------------------\n")
    return models.Model(inputs=[x], outputs=[out_cpxn])
    #'''
'''
model = CpXnNet(input_shape1=[28, 28, 1],n_class=10,epsilon = 1e-6,momentum = 0.9,
    use_bias = False, num_routing=3)

model.summary()

try:
    plot_model(model, to_file='./Without_Recon/Caps_structure.png', show_shapes=True, show_layer_names=True)
except Exception as e:
    print('No fancy plot {}'.format(e))
'''

def train(model, data, NB_EPOCHS, batch_size):

    # unpacking the data
    (x_train, y_train), (x_test, y_test) = data
    # learning rate schedule
    lr_start = 1e-3
    lr_end = 1e-6
    lr_decay = (lr_end / lr_start)**(1. / NB_EPOCHS)

    # callbacks
    log = callbacks.CSVLogger('./Without_Recon/Caps_log.csv')
    checkpoint = callbacks.ModelCheckpoint('./Without_Recon/Caps_weight.h5',
                                           save_best_only=True, 
                                           save_weights_only=True, verbose=1)
    lr_schdl = callbacks.LearningRateScheduler(schedule=lambda e: lr_start * lr_decay ** e)
    #lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: 0.0001 * np.exp(-epoch / 10.))
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
                  loss=[margin_loss], loss_weights=[1.0],
                  #loss=[margin_loss,'mse'],
                  #loss_weights=[1.0, 0.0005],
                  metrics={'out_cpxn':'accuracy'})
    #model.load_weights('trained_model.h5')
    #"""
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=NB_EPOCHS,
                        verbose=1, validation_data=[x_test, y_test], 
                        callbacks=[log, lr_schdl])
    
    """
    history = model.fit([x_train, y_train], [y_train, x_train], batch_size=batch_size, epochs=NB_EPOCHS,
                        verbose=1, validation_data=[[x_test, y_test], [y_test, x_test]], 
                        callbacks=[log, lr_schdl])
    """
    model.save_weights('./Without_Recon/Caps_weight.h5')
    print('Trained model saved to \'./Without_Recon/Caps_weight.h5\'')

    # dict_keys(['val_loss', 'val_acc', 'loss', 'acc', 'lr'])
    print("The list of record that can be plotted out:", history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    #plt.plot(history.history['out_cpxn_accuracy'])
    #plt.plot(history.history['val_out_cpxn_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='lower right')
    #plt.show()
    plt.savefig('./Without_Recon/Caps_accuracy.png')
    plt.clf()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    #plt.plot(history.history['out_cpxn_loss'])
    #plt.plot(history.history['val_out_cpxn_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    #plt.show()
    plt.savefig('./Without_Recon/Caps_loss.png')
    plt.clf()

    return model
"""
before_T = time.time()
train(model=model, data=((x_train, y_train), (x_test, y_test)), NB_EPOCHS=2, batch_size=150)
after_T = time.time()
"""    

def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(np.sqrt(num))
    height = int(np.ceil(float(num)/width))
    shape = generated_images.shape[1:3]
    image = np.zeros((height*shape[0], width*shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
            img[:, :, 0]
    return image

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

from sklearn.metrics import classification_report, confusion_matrix   
#def plott(a,b,x_test,x_recon): 
def plott(a,b,x_test): 
    label_dict = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4',
                  5: '5', 6: '6', 7: '7', 8: '8', 9: '9'}
    print(confusion_matrix(a, b)) 
    print(classification_report(a, b,
                                target_names=list(label_dict.values()),digits=3))
    '''
    img = combine_images(np.concatenate([x_test[:50],x_recon[:50]]))
    image = img * 255
    Image.fromarray(image.astype(np.uint8)).save("./Caps_Result/Caps_recon.png")
    print()
    print('Reconstructed images are saved to ./Caps_Result/Caps_recon.png')
    print('-'*50)
    plt.imshow(plt.imread("./Caps_Result/Caps_recon.png", ))
    plt.show()
    '''
#"""
with tf.Graph().as_default() as graph:
    model = CpXnNet(input_shape1=[28, 28, 1],n_class=10,epsilon = 1e-6,momentum = 0.9,
                    use_bias = False, num_routing=3)
    model.summary()
    try:
        plot_model(model, to_file='./Without_Recon/Caps_structure.png', show_shapes=True, show_layer_names=True)
    except Exception as e:
        print('No fancy plot {}'.format(e))

    before_T = time.time()
    train(model=model, data=((x_train, y_train), (x_test, y_test)),
          NB_EPOCHS=60, batch_size=150)
    after_T = time.time()
    a,b,x_test,top1,top5 = test(model=model, data=(x_test, y_test))
    plott(a,b,x_test) 
    stats_graph(graph)
print('-'*50)
print('Top1 Test Acc:', top1)
print('Top5 Test Acc:', top5)
print('-'*50)
print("The total training time is: ", (after_T-before_T), " seconds.")
#"""

'''
Total params: 1,691,616
Trainable params: 1,690,544
Non-trainable params: 1,072

The total training time is:  8502.187513589859  seconds.
8502.187513589859 / 60 = 141.70 seconds
acc: 0.9934

155636231

======================
Without Reconstruct
======================
1
Top1 Test Acc: 99.40\%
Top5 Test Acc: 99.96\%

--------------------------------------------------
FLOPs: 3889500;    Trainable params: 1690544
The total training time is:  5589.413856267929  seconds.
--------------------------------------------------

2
Top1 Test Acc: 99.44\%
Top5 Test Acc: 99.89\%

--------------------------------------------------
FLOPs: 3889500;    Trainable params: 1690544
The total training time is:  5839.49662899971  seconds.
--------------------------------------------------

3
Top1 Test Acc: 99.42\%
Top5 Test Acc: 100\%

--------------------------------------------------
FLOPs: 3889500;    Trainable params: 1690544
The total training time is:  6578.1999168396  seconds.
--------------------------------------------------

4
Top1 Test Acc: 99.38\%
Top5 Test Acc: 99.99\%

--------------------------------------------------
FLOPs: 3889500;    Trainable params: 1690544
The total training time is:  7208.7137904167175  seconds.
--------------------------------------------------

5
Top1 Test Acc: 99.42\%
Top5 Test Acc: 99.94\%

--------------------------------------------------
FLOPs: 3889500;    Trainable params: 1690544
The total training time is:  6790.938714265823  seconds.
--------------------------------------------------

top1    = [99.40,99.44,99.42,99.38,99.42]
top5    = [99.96,99.89,100,99.99,99.94]
time_ls = [5589.413856267929,5839.49662899971,6578.1999168396,7208.7137904167175,6790.938714265823]

Standard Deviation
Top1 99.412 0.020396078054371544
Top5 99.956 0.03929376540877598
Time 6401.352581357956 601.6329547847423

FLOPS: 10030266;    Trainable params1: 3928832
'''





