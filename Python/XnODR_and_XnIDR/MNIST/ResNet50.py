import os, re, time, json
import PIL.Image, PIL.ImageFont, PIL.ImageDraw
import numpy as np

import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from matplotlib import pyplot as plt

print("Tensorflow version " + tf.__version__)

BATCH_SIZE = 80
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
    input_images = input_images.astype('float32')
    input_images = np.stack((input_images,)*3, axis=-1)
    output_ims = tf.keras.applications.resnet50.preprocess_input(input_images)
    return output_ims

def feature_extractor(inputs):
    feature_extractor = tf.keras.applications.resnet.ResNet50(input_shape=(224, 224, 3),
                                               include_top=False,
                                               weights='imagenet')(inputs)
    return feature_extractor

def classifier(inputs):
    x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1024, activation="relu")(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.Dense(10, activation="softmax", name="classification")(x)
    return x

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
 
    model.compile(optimizer='SGD', 
                  loss='sparse_categorical_crossentropy',
                  metrics = ['accuracy'])
  
    return model

with tf.Graph().as_default() as graph:
    (training_images, training_labels) , (validation_images, validation_labels) = tf.keras.datasets.mnist.load_data()

    display_images(training_images, training_labels, training_labels, "Training Data" )
    display_images(validation_images, validation_labels, validation_labels, "Validation Data" )

    train_X = preprocess_image_input(training_images)
    valid_X = preprocess_image_input(validation_images)

    model = define_compile_model()

    model.summary()

    EPOCHS = 15
    t_start = time.time()
    history = model.fit(train_X, training_labels, epochs=EPOCHS, validation_data = (valid_X, validation_labels), batch_size=80)
    loss, accuracy = model.evaluate(valid_X, validation_labels, batch_size=80)
    t_end   = time.time()
    stats_graph(graph)
print("loss: ", loss)
print("accuracy: ", accuracy)
print("Running Time is: ", t_end-t_start)
