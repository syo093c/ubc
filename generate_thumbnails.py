import pandas as pd
import torch
from torch.utils.data import Dataset,dataloader
from PIL import Image,ImageFile
import numpy as np
import cv2
import ipdb
from glob import glob
from tqdm import tqdm
from rich.progress import track

DATA_ROOT='/home/syo/kaggle/input/UBC-OCEAN/'
MYIMG_DATA_ROOT=DATA_ROOT+'data/'
MYTRAIN_DATA_ROOT=MYIMG_DATA_ROOT+'train/'
TRAIN_IMG_ROOT=DATA_ROOT+'train_images/'
TEST_IMG_ROOT=DATA_ROOT+'test_images/'
TRAIN_THUMB_IMG_ROOT=DATA_ROOT+'train_thumbnails/'
TEST_THUMB_IMG_ROOT=DATA_ROOT+'test_thumbnails/'
TRAIN_CSV=DATA_ROOT+'train.csv'

def main():
    train_img_list=glob(TRAIN_IMG_ROOT+'*')
    new_width=4096
    Image.MAX_IMAGE_PIXELS = None

    for img in track(train_img_list):
        img_name=img.split('/')[-1]

        img=Image.open(img)
        width_percent = (new_width / float(img.size[0]))
        new_height = int((float(img.size[1]) * float(width_percent)))
        resized_img=img.resize((new_width,new_height),Image.LANCZOS)
        resized_img.save(MYTRAIN_DATA_ROOT+img_name)

if __name__ == '__main__':
    main()