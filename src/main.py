# %%
#libraries to import
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics  import categorical_crossentropy
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
train = '../data/train'
valid = '../data/valid'
test = '../data/test'

# %%
#larger class train/test split
train_trial = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input)\
    .flow_from_directory(directory=train, classes=['A','B','C', 'D','E','F','G','H','I','J','K','L','M','N','O','P','Q'] , batch_size=5)
# %
imgs, label = next(train_trial)
# %%
label
# %%
plotImages(imgs, 5)
# %%
test = tf.image.rgb_to_grayscale(imgs)
img, label
plotImages(, 5)

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



# %%

file_manager('blank','test/', small_class, 80)

# %%
test = get_imlist('../data/train/A','.jpg')
# %%
test
# %%
from skimage import io, color, filters
from skimage.color import rgb2gray
io.imshow('../data/train/A/A_63.jpg')

test2 = tf.keras.preprocessing.image.load_img('../data/train/A/A_63.jpg', color_mode='grayscale')
# %%
test2
# %%
type(test2)
# %%
