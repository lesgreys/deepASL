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
from sklearn.metrics import classification_report

import itertools
import os
import glob
import shutil
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

%matplotlib inline

#sudo code

# %%
def data_gen_object(path, batch_size, shuffle=True):
  """ 
  Takes care of preprocess, grayscale, augmentation and generator object
  Pass in the director for object generator and preprocess data with Keras API, ImageDataGenerator.
  Returns: DirectoryIterator from Keras API
  """
  dir_iter = tf.keras.preprocessing.image.ImageDataGenerator(image_dataset_from_directory(path,color_mode='grayscale', image_size=(100,100)), rescale=1./255)\
    .flow_from_directory(path, batch_size=batch_size, class_mode='categorical', shuffle=shuffle)

  return dir_iter

# %%
#building basic CNN

def build__compile_cnn():
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

def fit(model, train_data, epochs, validation_data):
  return model.fit(train_data,epochs=epochs,validation_data=validation_data)

def predict(model, test_data):
  y_pred = np.argmax(model.predict(test_data), axis=1)
  y_true = [np.argmax(test_data[i][1]) for i in range(747)]

  return y_pred, y_true


  
if __name__ == '__main__':
#establish paths for image processing using keras
  train_path = '../data/train'
  valid_path = '../data/valid'
  test_path = '../data/test'

  train = data_gen_object(train_path, 500)
  valid = data_gen_object(valid_path, 500)
  test = data_gen_object(test_path, 1, shuffle=False)

  model = build_cnn()
  

  cm = confusion_matrix(y_true,y_pred)

  plot_confusion_matrix(cm, test_gen.class_indices.keys())

# %%
