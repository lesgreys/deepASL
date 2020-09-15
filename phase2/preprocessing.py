
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

def fileName(df, label):

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
    return df

def pullURL(urlList, fNamelist, path=None):
    """
    save all videos locally into relative directory and assign unique filename

    Parameters:
        urlList (list/array): url of all videos to download
        fNamelist (list/array): unique file names for each video
        path (string): relative directory to save each video

    Returns:
        None
    """
    vidList = []
    for url, fname in zip(urlList, fNamelist):
        try:
            vid = YouTube(str(url)).streams.first().download(output_path=path, filename=fname)
        except:
            continue

list = ['https://www.youtube.com/watch?v=7YYB3BEoksc','https://www.youtube.com/watch?v=mC0lNJ6iz-s']

#function to pull in video
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