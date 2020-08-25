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
train_img_pp = image_dataset_from_directory(train_path,color_mode='grayscale', image_size=(100,100))
valid_img_pp = image_dataset_from_directory(valid_path,color_mode='grayscale', image_size=(100,100))
# test_img_pp = image_dataset_from_directory(test_path,color_mode='grayscale') (not needed yet.)

#create augmentation of image with ImageDataGenerator
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(train_img_pp, rescale=1./255)
valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(valid_img_pp, rescale=1./255)
# test_datagen =  tf.keras.preprocessing.image.ImageDataGenerator(test_img_pp)
# %%
#generator object
train_gen = train_datagen.flow_from_directory(train_path, 
batch_size=500, 
class_mode='categorical')

valid_gen = train_datagen.flow_from_directory(valid_path, 
batch_size=500, 
class_mode='categorical')


# %%
#building basic CNN
model = keras.models.Sequential()

model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(5,5)))
model.add(keras.layers.Dropout(.15))

model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(5,5)))
model.add(keras.layers.Dropout(.15))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(24, activation='sigmoid'))

# %%
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
# %%
model.fit(train_gen,epochs=5,validation_data=valid_gen)

# %%

# %%
def blank_():
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
pass 
# %%
# use current model to predict on test set

#generate test data
test_img_pp = image_dataset_from_directory(test_path,color_mode='grayscale', image_size=(100,100)) 
test_datagen =  tf.keras.preprocessing.image.ImageDataGenerator(test_img_pp, rescale=1./255)
test_gen = train_datagen.flow_from_directory(test_path, class_mode='categorical', shuffle=False, batch_size=1)
# %%
# test_loss, test_acc =  model.evaluate(test_gen)
# print('\nTest accuracy {:5.2f}%'.format(100*test_acc))
# %%
y_pred = np.argmax(model.predict(test_gen), axis=1)
y_predict = (model.predict(test_gen) > 0.5).astype("int32")
# %%

y_true = []
for i in range(747):
    y_true.append(np.argmax(test_gen[i][1]))



# %%
y_pred.shape, y_true.shape

# %%
# confusion_matrix(test_gen)
#converting num category to key category to see prediction of alphabet
# labels = (test_gen.class_indices)
# labels = dict((v,k)for k,v in labels.items())
# predictions = [labels[k] for k in y_pred]
# predictions

test_gen.reset()
pred= model.predict(test_gen)
predicted_class_indices=np.argmax(pred,axis=1)
labels=(test_gen.class_indices)
labels2=dict((v,k) for k,v in labels.items())
predictions=[labels2[k] for k in predicted_class_indices]
print(predicted_class_indices)
print(labels)
print(predictions)


# %%
cm = confusion_matrix(y_true,y_pred)
# %%
def plot_confusion_matrix(cm, class_names):

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
  return figure


# %%

plot_confusion_matrix(cm, test_gen.class_indices.keys())
plt.savefig('../images/confusion_matrix.png')
# %%

# %%
