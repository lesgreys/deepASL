
import json
import pandas as pd
import numpy as np

import urllib
from bs4 import BeautifulSoup
from pytube import YouTube
import cv2
import os
import glob
from tqdm import tqdm
import math


class dataLoader():
    pass

def dfmakeColumns(df):

    """
    create column for unique file names to save videos

    Parameters:
    df (DataFrame): pandas dataframe 
    label (string): name of column to pass in as unique file name

    Returns:
    New dataframe with new filename column
    """
    for i in range(len(df.index)):
        df['filename'][i] = str(df.index[i])+'-'+str(df.label[i])
        df['path'][i] = str(df.text)+'/'+str(df.index[i])+'-'+str(df.label[i])+'.mp4'
    return df


def pullURL(urlList, fNamelist, className, main_dir):
    """
    save all videos locally into relative directory and assign unique filename

    Parameters:
        urlList (list/array): url of all videos to download
        fNamelist (list/array): unique file names for each video
        class (list/array): class name for each subdirectory
        main_dir (string): main directory path where each video
    Returns:
        None
    """

    
    for url, fname, class_ in zip(urlList, fNamelist, className):
        os.chdir(str(main_dir))
        try:
            vid = YouTube(str(url)).streams.first()
            if os.path.isdir(str(class_)) is False:
                 os.makedirs(str(class_))
                 os.chdir(str(main_dir)+str(class_))
                 vid.download(filename=fname)
            else:    
                os.chdir(str(main_dir)+str(class_))
                vid.download(filename=fname)
        except:
            continue

def get_vidDirectory(main_path, end_char):
    """ 
    create list of all the end file directory for every class

    Parameters:
        main_path(string): main directory path for root folder
        end_char(string): desired file extension the file name ends

    Returns:
        list of all file directory
    """
    lst = []
    for class_ in os.listdir(str(main_path)):
        if class_.endswith('Store'):
            continue
        else:
            for f in os.listdir(f'{main_path}{class_}/'):
                if f.endswith(str(end_char)):
                    lst.append(os.path.join(f'{main_path}{class_}',f))
    return lst


def subClip(main_dir, df):
    #function for subclip; identifing if class is imbedded into larger video 
    #if yes process video through clip (moviepy library) to clip exact portion for class
        #parameters will be start_time, end_time, video_path
    #if no proceed extract frames from video
        #break
    """ 
    Identifies which vidoes require subclipping and cross-references with dataframe.
    Also deletes original video that has additional ASL. 

    Parameters:
        main_dir(string): main directory path for root folder where subfolders are all the classes
        df(pandas Dataframe): dataframe that holds all features of each video

    Returns:
        None
    """
    for class_ in os.listdir(str(main_dir)):
        if not class_.endswith('Store'):
            os.chdir(str(main_dir)+str(class_))
            !pwd
            for fname in os.listdir(str(main_dir)+str(class_)):
                if not fname.endswith('Store'):
                    x = df[df.filename == str(fname)]
                    start = x.start_time.values[0]
                    end = x.end_time.values[0]
                    if not start == 0:
                        VideoFileClip(str(os.path.join(f'{main_dir}{class_}',fname)), audio=False).subclip(float(start),float(end)).write_videofile('C-'+str(fname))
                        os.remove(str(fname))
                    else:
                        continue
                else:
                    continue
        else:
            continue
#function to pull in video COMPLETE
    #pass in url with youtube library

#append local path to dataframe

#function to organize each video with class name and subdirectory

#function for subclip; identifing if class is imbedded into larger video 
    #if yes process video through clip (moviepy library) to clip exact portion for class
        #parameters will be start_time, end_time, video_path
    #if no proceed extract frames from video
        #break

#function to extract frames from videos
    # parameters to pass in:
        #desired captured frame rate 
        #current frame rate
        #path of video 

if __name__ == __main__:
    pass