
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from skimage.io import imread
from glob import glob
import itertools

from skimage.io import imread
from skimage.color import rgb2gray


def ImagetoArray(list_img_paths):
    return np.array([rgb2gray(imread(i)) for i in list_img_paths])

def plotImages(images_arr, batch_size):

    fig, axes = plt.subplots(1,batch_size, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def plotImagesDist(images_arr, batch_size):
    plt.figure(figsize=(12,5))
    for img in images_arr:
        ax = sns.distplot(img)
    plt.tight_layout()
    plt.show()



def plot_samples(letter, random_num):
    print("Samples images for letter " + letter)
    base_path = 'data/train/'
    img_path = base_path + letter + '/**'
    path_contents = glob(img_path)
    
    plt.figure(figsize=(5,5))
    imgs = random.sample(path_contents, random_num)
    plt.subplot(151)
    plt.axis('off')
    plt.imshow(imread(imgs[0]))
    plt.subplot(152)
    plt.axis('off')
    plt.imshow(imread(imgs[1]))
    plt.subplot(153)
    plt.axis('off')
    plt.imshow(imread(imgs[2]))
    plt.subplot(154)
    plt.axis('off')
    plt.imshow(imread(imgs[3]))
    plt.subplot(155)
    plt.axis('off')
    plt.imshow(imread(imgs[4]))
    print(imgs)
    return


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
  plt.show()
  return figure

def plotMissedClasses(images_path, predicted_values, true_values):
  """
  Plots images model wrongly predicted:

  Parameters:
  images_path (array/list): each element is a string of the path for each image predicted on.
  predicted_values (array/list): each element is the class model predicted.
  true_values (array/list): each element is the true class of the image. 

  Returns:
  Plots and saves each image with True Class & Predicted Class identified. 
  """
  image_array = ImagetoArray(images_path) #creates an array of arrays where each image path is converted to grayscale number array. 
  for num, (img, pred, true) in enumerate(zip(image_array, predicted_values, true_values)):
      fig, ax = plt.subplots()
      ax.set_title(f'Class: {true}, Predicted: {pred}')
      ax.imshow(img)
      ax.axis('off')
      plt.savefig(f'../images/{true}{pred}{num}.png')
