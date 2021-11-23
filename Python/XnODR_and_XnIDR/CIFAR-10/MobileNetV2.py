import os, re, time, json
import PIL.Image, PIL.ImageFont, PIL.ImageDraw
import numpy as np

import tensorflow as tf
from clr_callback import *
from matplotlib import pyplot as plt
from keras.utils.vis_utils import plot_model
from keras import callbacks

#K.set_image_data_format('channels_last')  # for capsule net
#K.clear_session()
print("Tensorflow version " + tf.__version__)

BATCH_SIZE = 100
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
    output_ims = tf.keras.applications.mobilenet_v2.preprocess_input(input_images)
    return output_ims

def feature_extractor(inputs):
    #feature_extractor = tf.keras.applications.mobilenet_v2.MobileNetV2(input_shape=(224, 224, 3),
    #                                           include_top=False, weights='imagenet')(inputs)
    feature_extractor = tf.keras.applications.mobilenet_v2.MobileNetV2(input_shape=(224, 224, 3),
                                               include_top=False)(inputs)
    return feature_extractor

def classifier(inputs):
    x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(512,activation=('relu'))(x)
    x = tf.keras.layers.Dense(256,activation=('relu'))(x)
    x = tf.keras.layers.Dropout(.3)(x)
    x = tf.keras.layers.Dense(128,activation=('relu'))(x)
    x = tf.keras.layers.Dropout(.2)(x)
    x = tf.keras.layers.Dense(10,activation=('softmax'))(x)
    return x

def final_model(inputs):

    resize = tf.keras.layers.UpSampling2D(size=(7,7))(inputs)

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

    train_X = preprocess_image_input(training_images)
    valid_X = preprocess_image_input(validation_images)

    model = define_compile_model()

    model.summary()

    plot_model(model, show_shapes=True, show_layer_names=True)
    try:
        plot_model(model, to_file='./Log_Mbl/MBV2O_structure.png', show_shapes=True, show_layer_names=True)
    except Exception as e:
        print('No fancy plot {}'.format(e))
    EPOCHS = 80
    log = callbacks.CSVLogger('./Log_Mbl/MBV2O_log.csv')
    checkpoint = callbacks.ModelCheckpoint('./Log_Mbl/MBV2O_weight.h5',
                                           save_best_only=True, verbose=1,
                                           monitor='val_accuracy', mode='max',
                                           save_weights_only=True)
    lr_schdl = CyclicLR(mode='triangular2')
    # learning rate schedule
    lr_start = 1e-3
    lr_end = 1e-6
    lr_decay = (lr_end / lr_start)**(1. / EPOCHS)
    #lr_schdl = callbacks.LearningRateScheduler(schedule=lambda e: lr_start * lr_decay ** e)
    
    #model.load_weights('./Log_Mbl/MBV2O_weight.h5')
    
    history = model.fit(train_X, training_labels, epochs=EPOCHS, 
        validation_data = (valid_X, validation_labels), batch_size=100,
        callbacks=[log, checkpoint])
    print("The list of record that can be plotted out:", history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='lower right')
    plt.savefig('./Log_Mbl/MBV2O_accuracy.png')
    plt.clf()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.savefig('./Log_Mbl/MBV2O_loss.png')
    plt.clf()
    #, lr_schdl
    #model.save_weights('./Log_Mbl/MBV2Cyc_weight.h5')
    before_T = time.time()
    loss, accuracy = model.evaluate(valid_X, validation_labels, batch_size=100)
    after_T = time.time()
    stats_graph(graph)
print("loss: ", loss)
print("accuracy: ", accuracy)
print("Running Time: ", after_T-before_T)

"""
FLOPs: 6157200;    Trainable params: 3045258
loss:  0.35255646912381056accuracy:  0.9462
Running Time:  8821.123060703278

1
FLOPs: 6157200;    Trainable params: 3045258
loss:  0.37325499933212997
accuracy:  0.9291
Running Time:  3714.943257331848

2
FLOPs: 6157200;    Trainable params: 3045258
loss:  0.3625707419589162
accuracy:  0.9333
Running Time:  4848.5132830142975

3
FLOPs: 6157200;    Trainable params: 3045258
loss:  0.36256603349000216
accuracy:  0.9312
Running Time:  5028.965527772903

4
FLOPs: 6157200;    Trainable params: 3045258
loss:  0.36001460524275897
accuracy:  0.9348
Running Time:  4920.658530950546

5
FLOPs: 6157200;    Trainable params: 3045258
loss:  0.3333879154548049
accuracy:  0.9369
Running Time:  4815.19681429863

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

FLOPs: 6157200;    Trainable params: 3045258
loss:  0.389070438221097
accuracy:  0.9367
Running Time:  8775.76508975029
"""
