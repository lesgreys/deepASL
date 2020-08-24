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
def file_manager(main_dir, create_path, class_list, num_random_files):
    """organizes files into desired directory structure for train/valid/test"""

    os.chdir(str(main_dir))
    for class_ in class_list:
        if os.path.isdir(str(create_path)+str(class_)) is False:
            os.makedirs(str(create_path)+str(class_))


            for c in random.sample(glob.glob(str(class_)+'*'), num_random_files):
                shutil.move(c, str(create_path)+str(class_))

def img_load():
    pass

def plotImages(images_arr, batch_size):
    fig, axes = plt.subplots(1,batch_size, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def get_imlist(path,end_char):
  """  Returns a list of filenames for
    all png images in a directory. """

  return [os.path.join(path,f) for f in os.listdir(path) if f.endswith(str(end_char))]

def get_classlist(path):
    """Returns a list of directories for each class"""


def convert_grayscale(path):
    pass

# %%
#establish paths for image processing using keras
train_path = '../data/train'
valid_path = '../data/valid'
test_path = '../data/test'

# %%

# image prepreprocessing from rgb to grayscale.. play around with other parameters to see impact of changes
train_img_pp = image_dataset_from_directory(train_path,color_mode='grayscale')
valid_img_pp = image_dataset_from_directory(valid_path,color_mode='grayscale')
# test_img_pp = image_dataset_from_directory(test_path,color_mode='grayscale') (not needed yet.)

#create augmentation of image with ImageDataGenerator
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(train_img_pp)
valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(valid_img_pp)
# test_datagen =  tf.keras.preprocessing.image.ImageDataGenerator(test_img_pp)
# %%
#generator object
train_gen = train_datagen.flow_from_directory(train_path, 
batch_size=100, 
class_mode='categorical')

valid_gen = train_datagen.flow_from_directory(valid_path, 
batch_size=100, 
class_mode='categorical')


# %%
#building basic CNN
model = keras.models.Sequential()
model.add(keras.layers.Conv2D(32, (5, 5), activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))

model.add(keras.layers.Conv2D(32, (5, 5), activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))

model.add(keras.layers.Conv2D(64, (5, 5), activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(24, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(24,
                activation='tanh'))

# %%
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
# %%
model.fit(
    train_gen,
    epochs=2,
    validation_data=valid_gen)


# %%
#from A-Q data set is 400 observations 'large_class', train, valid, test = [320,40,40]
#from S-Y data set is 100 observations 'small_class', train, valid, test = [80,10,10]

# P & Q seem to be different from ASL, Z is not included because it's a dynamic sign
# J shouldn't be included but is signed differently from 
# found that X had 23 miss classified files from D but labeled X (train, valid, test set = 57, 7, 7)


class_list = ['A','B','C', 'D','E','F','G','H','I','J','K','L','M','N','O','P','Q','S','T','U','V','W','X','Y']
create_path = ['train/','valid/','test/']
num_random_files = {'large_class': [320,40,40], 'small_class':[80,10,10]}
main_dir = '/data/Train'

file_manager(main_dir, create_path, small_class, num_random_files)

