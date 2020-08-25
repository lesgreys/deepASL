
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from skimage.io import imread
from glob import glob


def plotImages(images_arr, batch_size):
    fig, axes = plt.subplots(1,batch_size, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def plotImagesDist(images_arr, batch_size):
    plt.figure(figsize=(8,3))
    for img in images_arr:
        ax = sns.distplot(img)
    plt.tight_layout()
    plt.show()



def plot_samples(letter, random_num):
    print("Samples images for letter " + letter)
    base_path = '../data/train/'
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