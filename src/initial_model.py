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

def build_cnn():
    model = keras.models.Sequential()

    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(5,5)))
    model.add(keras.layers.Dropout(.15))

    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(5,5)))
    model.add(keras.layers.Dropout(.15))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(24, activation='sigmoid'))

    return model


# %%
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
# %%
model.fit(train_gen,epochs=5,validation_data=valid_gen)

# %
# %%
# use current model to predict on test set

#generate test data
test_img_pp = image_dataset_from_directory(test_path,color_mode='grayscale', image_size=(100,100)) 
test_datagen =  tf.keras.preprocessing.image.ImageDataGenerator(test_img_pp, rescale=1./255)
test_gen = train_datagen.flow_from_directory(test_path, class_mode='categorical', shuffle=False, batch_size=1)


y_pred = np.argmax(model.predict(test_gen), axis=1)
y_predict = (model.predict(test_gen) > 0.5).astype("int32")
# %%

y_true = [np.argmax(test_gen[i][1]) for i in range(747)]
y_true


# %%
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

plot_confusion_matrix(cm, test_gen.class_indices.keys())
# plt.savefig('../images/confusion_matrix.png')
#
# %%


if __name__ == '__main__':
  #establish paths for image processing using keras
  train_path = '../data/train'
  valid_path = '../data/valid'
  test_path = '../data/test'

