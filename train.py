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

from dataloader import build_dataset_dataloader
from model import UBCModel
from lightning.pytorch.loggers import WandbLogger

from lightning.pytorch.callbacks import LearningRateMonitor
import torchvision
import torch

from lightning.pytorch.callbacks import ModelCheckpoint

def build_med_resnet():
    weight = torch.load("sadf.torch")
    model = torchvision.models.resnet50(pretrained=True)
    model.fc = torch.nn.Linear(in_features=2048, out_features=5, bias=True)
    model.load_state_dict(weight, strict=False)
    return model


def build_convnext():
    fn = model_entrypoint("convnext_small.in12k_ft_in1k_384")
    cfg = {
        "num_classes": 5,
        #"pretrained": False,
        "pretrained": True,
        # input_size=(3, 384, 384),
        #'input_size':(3,1024,1024),
    }
    convnext_s = fn(**cfg)
    return convnext_s


def main():
    train_dataset, train_dataloader = build_dataset_dataloader(
        #batch_size=16, type="train"
        batch_size=8, type="train"
    )
    validation_dataset, validation_dataloader = build_dataset_dataloader(
        type="validation"
    )

    model = UBCModel(
        #model=build_med_resnet(),
        model=build_convnext(),
        train_dataloader=train_dataloader,
        valid_dataloader=validation_dataloader,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")
    wandb_logger = WandbLogger(project="ubc", name="baseline2_w-valid_w-data-med_resnet")
    checkpoint_callback = ModelCheckpoint(monitor='train/valid_loss',save_top_k=3)
    #checkpoint_callback = ModelCheckpoint(monitor='train/valid_scores',save_top_k=1)
    #trainer = L.Trainer( max_epochs=40, precision="bf16", logger=wandb_logger, callbacks=[lr_monitor],log_every_n_steps=1)
    trainer = L.Trainer( max_epochs=30, precision="bf16", logger=wandb_logger, callbacks=[lr_monitor,checkpoint_callback],log_every_n_steps=1)
    #trainer = L.Trainer(max_epochs=100,precision='bf16',callbacks=[lr_monitor])
    # trainer = L.Trainer(max_epochs=800,gradient_clip_val=0.5,precision='bf16',logger=wandb_logger,callbacks=[lr_monitor],log_every_n_steps=50)
    # trainer = L.Trainer(max_epochs=100,precision='bf16')
    trainer.fit( model=model, train_dataloaders=train_dataloader, val_dataloaders=validation_dataloader,)


if __name__ == "__main__":
    main()
