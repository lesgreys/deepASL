# %%
#libraries to import
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics  import categorical_crossentropy
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
from plot_helper import plot_confusion_matrix

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def data_gen_object(path, batch_size, shuffle=True):
  """ 
  Takes care of preprocess, grayscale, augmentation and generator object
  Pass in the director for object generator and preprocess data with Keras API, ImageDataGenerator.
  Returns: DirectoryIterator from Keras API
  """
  dir_iter = tf.keras.preprocessing.image.ImageDataGenerator(image_dataset_from_directory(path,color_mode='grayscale', image_size=(100,100)), rescale=1./255)\
    .flow_from_directory(path, batch_size=batch_size, class_mode='categorical', shuffle=shuffle)

  return dir_iter

def data_aug_object(path, batch_size, shuffle=True):
  """ 
  Takes care of preprocess, grayscale, augmentation and generator object
  Pass in the director for object generator and preprocess data with Keras API, ImageDataGenerator.
  Returns: DirectoryIterator from Keras API
  """
  dir_iter = tf.keras.preprocessing.image.ImageDataGenerator(
    image_dataset_from_directory(path,color_mode='grayscale', image_size=(100,100)), rescale=1./255,rotation_range=10,zoom_range=0.05,width_shift_range=0.05, height_shift_range=0.05).flow_from_directory(path, batch_size=batch_size, class_mode='categorical', shuffle=shuffle)
  return dir_iter

#building basic CNN

def build_compile_cnn():
    model = keras.models.Sequential()

    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(5,5)))
    model.add(keras.layers.Dropout(.15))

    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(5,5)))
    model.add(keras.layers.Dropout(.15))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(24, activation='sigmoid'))

    model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

    return model

# def fit(model, train_data, epoch, valid_data):
#   return model.fit(train_data,epochs=epochs,validation_data=valid_data)

def predict(model, test_data, test_observations=747):
  y_pred = np.argmax(model.predict(test_data), axis=1)
  y_true = np.array([np.argmax(test_data[i][1]) for i in range(test_observations)])

  return y_pred, y_true


if __name__ == '__main__':
#establish paths for image processing using keras
  train_path = '../data/train'
  valid_path = '../data/valid'
  test_path = '../data/test'
  epochs = 5
  batch_size=500

  train = data_aug_object(train_path, batch_size)
  valid = data_gen_object(valid_path, batch_size)
  test = data_gen_object(test_path, 1, shuffle=False)

  init_model = build_compile_cnn()
  init_model.fit(train,epochs=epochs,validation_data=valid)


  score = init_model.evaluate(test, verbose=0)
  print('Test score:', score[0])
  print('Test accuracy:', score[1])

  y_pred, y_true = predict(init_model, test)

  cm = confusion_matrix(y_true,y_pred)

  plot_confusion_matrix(cm, test.class_indices.keys(), 'confusion_matrix_sandbox')

  

# %%
