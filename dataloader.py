import pandas as pd
import torch
from torch.utils.data import Dataset,DataLoader
from PIL import Image
import numpy as np
import cv2
import ipdb
import os
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch import nn
import torch

import pytorch_lightning as pl
import timm
from timm.models.registry import model_entrypoint
import pickle
import torch.nn.functional as F

DATA_ROOT='/home/syo/kaggle/input/UBC-OCEAN/'
TRAIN_IMG_ROOT=DATA_ROOT+'train_images/'
TEST_IMG_ROOT=DATA_ROOT+'test_images/'
TRAIN_THUMB_IMG_ROOT=DATA_ROOT+'train_thumbnails/'
TEST_THUMB_IMG_ROOT=DATA_ROOT+'test_thumbnails/'
TRAIN_CSV=DATA_ROOT+'train.csv'

def get_cropped_images(file_path, image_id, th_area = 1000):
    image = Image.open(file_path)
    # Aspect ratio
    as_ratio = image.size[0] / image.size[1]
    
    sxs, exs, sys, eys = [],[],[],[]
    """
    如果这个图片的横向特别长,则进行切片操作。
    否则，直接整张图padding，resize送入。
    切片时，首先寻找联通分量。如果联通分量数目小于比例，说明是均等分布，均等切片。
    如果联通分量数目大于比例，说明有一些非常小的分量。考虑是否保留。

    TODO:如果这个图是一个纵向非常长的图呢。
    """
    if as_ratio >= 1.5:
        # Crop
        mask = np.max( np.array(image) > 0, axis=-1 ).astype(np.uint8)
        retval, labels = cv2.connectedComponents(mask)
        if retval >= as_ratio:
            x, y = np.meshgrid( np.arange(image.size[0]), np.arange(image.size[1]) )
            for label in range(1, retval):
                area = np.sum(labels == label)
                if area < th_area:
                    continue
                xs, ys= x[ labels == label ], y[ labels == label ]
                sx, ex = np.min(xs), np.max(xs)
                cx = (sx + ex) // 2
                crop_size = image.size[1]
                sx = max(0, cx-crop_size//2)
                ex = min(sx + crop_size - 1, image.size[0]-1)
                sx = ex - crop_size + 1
                sy, ey = 0, image.size[1]-1
                sxs.append(sx)
                exs.append(ex)
                sys.append(sy)
                eys.append(ey)
        else:
            #切成多个正方形的切片
            crop_size = image.size[1]
            for i in range(int(as_ratio)):
                sxs.append( i * crop_size )
                exs.append( (i+1) * crop_size - 1 )
                sys.append( 0 )
                eys.append( crop_size - 1 )
    else:
        # Not Crop (entire image)
        sxs, exs, sys, eys = [0,],[image.size[0]-1],[0,],[image.size[1]-1]

    df_crop = pd.DataFrame()
    df_crop["image_id"] = [image_id] * len(sxs)
    df_crop["file_path"] = [file_path] * len(sxs)
    df_crop["sx"] = sxs
    df_crop["ex"] = exs
    df_crop["sy"] = sys
    df_crop["ey"] = eys
    return df_crop

def get_test_file_path(image_id):
    if os.path.exists(f"{TEST_DIR}/{image_id}_thumbnail.png"):
        return f"{TEST_DIR}/{image_id}_thumbnail.png"
    else:
        return f"{ALT_TEST_DIR}/{image_id}.png"

def get_train_file_path(image_id):
    if os.path.exists(f"{TRAIN_THUMB_IMG_ROOT}/{image_id}_thumbnail.png"):
        return f"{TRAIN_THUMB_IMG_ROOT}/{image_id}_thumbnail.png"
    else:
        return f"{TRAIN_IMG_ROOT}/{image_id}.png"

class UBCDataset(Dataset):
    def __init__(self, df, raw_df, transforms=None):
        self.df = df
        self.raw_df=raw_df
        self.class_name=['CC', 'EC', 'HGSC', 'LGSC', 'MC']
        self.file_names = df['file_path'].values
        self.labels = df['label'].values
        self.transforms = transforms
        self.sxs = df["sx"].values
        self.exs = df["ex"].values
        self.sys = df["sy"].values
        self.eys = df["ey"].values
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        img_path = self.file_names[index]
        sx = self.sxs[index]
        ex = self.exs[index]
        sy = self.sys[index]
        ey = self.eys[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label_name = self.labels[index]
        label=self.class_name.index(label_name)
        label_onehot=F.one_hot(torch.tensor(label,dtype=torch.long),num_classes=len(self.class_name)).to(dtype=torch.float)
        
        img = img[ sy:ey, sx:ex, : ]
        
        if self.transforms:
            img = self.transforms(image=img)["image"]
            img=img.to(dtype=torch.float)
            
        return {
            'data': img,
            'label': label_onehot
        }

def build_data_loader(batch_size=4):
    CONFIG = {
    "seed": 42,
    "img_size":1024,
    "model_name": "tf_efficientnetv2_s_in21ft1k",
    "num_classes": 5,
    "valid_batch_size":4,
    "train_batch_size":4,
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
}
    if os.path.exists('debug_data'):
        with open('debug_data', 'rb') as f:
            df_crop=pickle.load(f)
    else:
        #======read train df====================
        train_df=pd.read_csv(TRAIN_CSV)
        train_df["file_path"] = train_df["image_id"].apply(get_train_file_path)
        
        #=======crop============================
        df=train_df
        dfs = []
        for (file_path, image_id) in tqdm(zip(df["file_path"], df["image_id"]),total=len(df)):
            dfs.append( get_cropped_images(file_path, image_id) )

        df_crop = pd.concat(dfs)
        df_crop=df_crop.merge(train_df[['image_id','label']],how='left',on='image_id')
        with open('debug_data', 'wb') as f:
            pickle.dump(df_crop, f)

    #===data augmentation==========================
    data_transforms = {
    "valid": A.Compose([
        A.Resize(CONFIG['img_size'], CONFIG['img_size']),
        A.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225], 
                max_pixel_value=255.0, 
                p=1.0
            ),
        ToTensorV2()], p=1.),
    "train": A.Compose([
        A.Resize(CONFIG['img_size'], CONFIG['img_size']),
        A.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225], 
                max_pixel_value=255.0, 
                p=1.0
            ),
        ToTensorV2()], p=1.),
    "dummy": A.Compose([
        A.Resize(CONFIG['img_size'], CONFIG['img_size']),
        ToTensorV2()],
        p=1.),
    }
    train_dataset=UBCDataset(df_crop,transforms=data_transforms["dummy"])
    train_datalodaer = DataLoader(train_dataset, batch_size=batch_size,
                          num_workers=2, shuffle=False, pin_memory=True)
    return train_datalodaer


def build_dataset_dataloader(batch_size=4,type='train'):
    CONFIG = {
    "seed": 42,
    "img_size":1024,
    "num_classes": 5,
}
    #======read train df====================
    df=pd.read_csv(TRAIN_CSV)
    
    #train_split=df.sample(n=int(0.8*len(df)), random_state=42)
    train_split=pd.concat([b.sample(n=int(0.9*len(b)), random_state=42) for a,b in df.groupby('label')])
    validation_mask = ~df.index.isin(train_split.index)
    valid_split = df[validation_mask]

    augmentation='dummy'
    if type=='train':
        df=train_split
        shuffle=True
        augmentation='train'
    elif type=='validation':
        df=valid_split
        batch_size=1
        shuffle=False

    df["file_path"] = df["image_id"].apply(get_train_file_path)
    
    #=======crop============================
    dfs = []
    for (file_path, image_id) in tqdm(zip(df["file_path"], df["image_id"]),total=len(df)):
        dfs.append( get_cropped_images(file_path, image_id) )

    df_crop = pd.concat(dfs)
    df_crop=df_crop.merge(df[['image_id','label']],how='left',on='image_id')

    #===data augmentation==========================
    data_transforms = {
    "dummy": A.Compose([
        A.Resize(CONFIG['img_size'], CONFIG['img_size']),
        A.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225),p=1.),
        ToTensorV2()],
        p=1.),
    "train":
    A.Compose([
        A.Resize(1024, 1024),
        #A.Downscale(scale_min=0.5,scale_max=0.9,p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.75),
        A.ShiftScaleRotate(p=0.75),
        A.OneOf([
            A.GaussNoise(var_limit=[10, 50]),
            A.GaussianBlur(),
            A.MotionBlur(),
            ], p=0.4),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
        A.CoarseDropout(max_holes=1, max_width=int(512* 0.3), max_height=int(512* 0.3),
                        mask_fill_value=0, p=0.5),
        A.RandomCrop(height=640,width=640,p=0.5),
        A.Resize(1024, 1024),
        A.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225),p=1.),
        ToTensorV2(),
        ],p=1.),
    }
    dataset=UBCDataset(df_crop,raw_df=df,transforms=data_transforms[augmentation])
    dataloader = DataLoader(dataset, batch_size=batch_size,
                          num_workers=2, shuffle=shuffle, pin_memory=True)
    return dataset,dataloader

if __name__ =='__main__':
    train_dataset, train_dataloader = build_dataset_dataloader(
        #batch_size=16, type="train"
        batch_size=8, type="train"
    )
    validation_dataset, validation_dataloader = build_dataset_dataloader(
        type="validation"
    )