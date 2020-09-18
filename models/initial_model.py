# %%


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics  import categorical_crossentropy
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
# from utils.plot_helper import plot_confusion_matrix
from tensorflow.keras.callbacks import History
import itertools

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
  dir_iter = tf.keras.preprocessing.image.ImageDataGenerator(image_dataset_from_directory(path, color_mode='rgb', image_size=(100,100)), rescale=1./255)\
    .flow_from_directory(path, batch_size=batch_size, class_mode='categorical', shuffle=shuffle)

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
    model.add(keras.layers.Dense(4, activation='softmax'))

    model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

    return model

# def fit(model, train_data, epoch, valid_data):
#   return model.fit(train_data,epochs=epochs,validation_data=valid_data)

def predict(model, test_data):
  y_pred = np.argmax(model.predict(test_data), axis=1)
  y_true = np.array([np.argmax(test_data[i][1]) for i in range(len(test_data))])

  return y_pred, y_true

def plot_confusion_matrix(cm, class_names, file_name):

  """
  Returns a matplotlib figure containing the plotted confusion matrix.

  Args:
    cm (array, shape = [n, n]): a confusion matrix of integer classes
    class_names (array, shape = [n]): String names of the integer classes
  """
  figure = plt.figure(figsize=(15, 15))
  plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
  plt.title("Confusion matrix")
  plt.colorbar()
  tick_marks = np.arange(len(class_names))
  plt.xticks(tick_marks, class_names, rotation=45)
  plt.yticks(tick_marks, class_names)

  # Normalize the confusion matrix.
  cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

  # Use white text if squares are dark; otherwise black.
  threshold = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    color = "white" if cm[i, j] > threshold else "black"
    plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.savefig(f'../images/{file_name}.png')
  return figure


if __name__ == '__main__':
#establish paths for image processing using keras
  train_path = '/home/ubuntu/000_homebase/deepASL/dataset/train'
  valid_path = '/home/ubuntu/000_homebase/deepASL/dataset/valid'
  test_path = '/home/ubuntu/000_homebase/deepASL/dataset/test'
  epochs = 1
  batch_size=100

  train = data_gen_object(train_path, batch_size)
  valid = data_gen_object(valid_path, batch_size)
  test = data_gen_object(test_path, 1, shuffle=False)

  init_model = build_compile_cnn()
  init_model.fit(train,epochs=epochs,validation_data=valid)

  init_model.save('init_model.h5')

  score = init_model.evaluate(test, verbose=1)
  print('Test score:', score[0])
  print('Test accuracy:', score[1])

  y_pred, y_true = predict(init_model, test)

  cm = confusion_matrix(y_true,y_pred)

  plot_confusion_matrix(cm, test.class_indices.keys(), 'confusion_matrix4')
