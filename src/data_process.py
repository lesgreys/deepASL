
import itertools
from skimage.io import imread
from skimage.color import rgb2gray
import os
import glob
import shutil
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)



def file_manager(main_dir, create_path, class_list, num_random_files):
    """organizes files into desired directory structure for train/valid/test"""

    os.chdir(str(main_dir))
    for class_ in class_list:
        if os.path.isdir(str(create_path)+str(class_)) is False:
            os.makedirs(str(create_path)+str(class_))


            for c in random.sample(glob.glob(str(class_)+'*'), num_random_files):
                shutil.move(c, str(create_path)+str(class_))



def get_imlist(path,end_char):
  """  Returns a list of filenames for
    all png images in a directory. """

  return [os.path.join(path,f) for f in os.listdir(path) if f.endswith(str(end_char))]
    
def arrMissedClass(main_dir, test_directory, test_true, test_predicted, class_dict):
  #get y_true array compare with y_pred array and return array of indexes 
  #where all pred != true.
    labels = dict((v,k) for k,v in class_dict.items())
    pred_arr = np.array([labels[k] for k in test_predicted])
    true_arr = np.array([labels[k] for k in test_true])

    #cross return array with array of test directory
    missclassdir = np.array([str(main_dir) + test_directory[i] for i in np.argwhere(test_true!=test_predicted).flatten()])
    misspredarr = np.array([pred_arr[i] for i in np.argwhere(test_true!=test_predicted).flatten()])
    misstruearr = np.array([true_arr[i] for i in np.argwhere(test_true!=test_predicted).flatten()])
    return missclassdir, misspredarr, misstruearr

if __name__ == '__main__':

    """
    initial variables set to structure folders 

    class_list = ['A','B','C', 'D','E','F','G','H','I','J','K','L','M','N','O','P','Q','S','T','U','V','W','X','Y']
    create_path = ['train/','valid/','test/']
    num_random_files = {'large_class': [320,40,40], 'small_class':[80,10,10]}
    main_dir = '/data/Train'

    file_manager(main_dir, create_path, small_class, num_random_files)
    """
    

    """
    #from A-Q data set is 400 observations 'large_class', train, valid, test = [320,40,40]
        #from S-Y data set is 100 observations 'small_class', train, valid, test = [80,10,10]

        # P & Q seem to be different from ASL, Z is not included because it's a dynamic sign
        # J shouldn't be included but is signed differently from 
        # found that X had 23 miss classified files from D but labeled X (train, valid, test set = 57, 7, 7)
    """

    pass