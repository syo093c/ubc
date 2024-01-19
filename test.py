import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
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

import lightning as L
import timm
from timm.models.registry import model_entrypoint
import pickle
import torch.nn.functional as F

from dataloader import build_test_data_loader,build_all_data
from model import ConvNext_S
from lightning.pytorch.loggers import WandbLogger

from lightning.pytorch.callbacks import LearningRateMonitor
from eval import score


def main():
    # init model
    model = ConvNext_S.load_from_checkpoint(
        "./ubc/baseline1_fix-lr_ziwbreee_softmax.ckpt", train_dataloader=None
    )
    model.eval()
    model = model.cuda()
    # load weight

    # test data
    test_dataset = build_test_data_loader()

    preds = []
    with torch.no_grad():
        bar = tqdm(enumerate(test_dataset), total=len(test_dataset))
        for step, data in bar:
            data = data["data"].unsqueeze(0).cuda()
            outputs = model(data)
            outputs = F.softmax(outputs)
            preds.append(outputs.detach().cpu().numpy())
    preds = np.vstack(preds)
    print(preds.shape)

    df_crop=test_dataset.df
    for i in range(preds.shape[-1]):
        df_crop[f"cat{i}"] = preds[:, i]

    dict_label = {}
    for image_id, gdf in df_crop.groupby("image_id"):
        dict_label[image_id] = np.argmax( gdf[ [f"cat{i}" for i in range(preds.shape[-1])] ].values.max(axis=0) )
        #dict_label[image_id] = np.argmax( gdf[ [f"cat{i}" for i in range(preds.shape[-1])] ].values.mean(axis=0) )
    preds = np.array( [ dict_label[image_id] for image_id in df_crop["image_id"].unique() ] )
    pred_labels=[test_dataset.class_name[i] for i in preds]
    gt_df=build_all_data()
    test_df=build_all_data()
    test_df["label"] = pred_labels
    # visualization
    scores=score(solution=gt_df[['image_id','label']],submission=test_df[['image_id','label']],row_id_column_name='image_id')
    print(scores)


if __name__ == "__main__":
    main()
