from typing import Any
import lightning as L
from pytorch_lightning.utilities.types import OptimizerLRScheduler
import torch
from torch import nn
import timm
from timm.models.registry import model_entrypoint
from torch import optim
from transformers import get_cosine_schedule_with_warmup
import ipdb
from eval import score
import torch.nn.functional as F
import numpy as np


class UBCModel(L.LightningModule):
    def __init__(self, model, train_dataloader, valid_dataloader) -> None:
        super().__init__()
        # fn = model_entrypoint("convnext_small.in12k_ft_in1k_384")
        # cfg = {
        #    "num_classes": 5,
        #    "pretrained": True,
        #    # input_size=(3, 384, 384),
        #    #'input_size':(3,1024,1024),
        # }
        # self.model = fn(**cfg)
        self.model = model
        # self.loss_fn=nn.BCEWithLogitsLoss()
        self.loss_fn = nn.CrossEntropyLoss()
        self.train_dl = train_dataloader
        self.train_ds = train_dataloader.dataset
        self.valid_ds = valid_dataloader.dataset

    def forward(self, input):
        output = self.model(input)
        return output

    def training_step(self, i):
        input = i["data"]
        label = i["label"]
        output = self.forward(input)
        loss = self.loss_fn(input=output, target=label)
        self.log("train/loss", loss)
        # self.log("lr",self.lr_schedulers().get_lr()[0])
        return loss

    def configure_optimizers(self):
        steps_per_ep = len(self.train_dl)
        train_steps = len(self.train_dl) * self.trainer.max_epochs  # max epouch 100
        #optimizer = optim.AdamW(self.parameters(), lr=1e-5)
        optimizer = optim.AdamW(self.parameters(), lr=1e-5)
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(steps_per_ep * self.trainer.max_epochs * 0.05),
            num_training_steps=train_steps,
        )
        return [optimizer], [
            {"scheduler": lr_scheduler, "interval": "step", "frequency": 1}
        ]

    def validation_step(self, i):
        input = i["data"]
        label = i["label"]
        output = self.forward(input)
        loss = self.loss_fn(input=output, target=label)
        self.log("train/valid_loss", loss)

    def get_score(self, type="valid"):
        preds = []
        if type == "valid":
            ds = self.valid_ds
        elif type == "train":
            ds = self.train_ds

        with torch.no_grad():
            for data in ds:
                data = data["data"].unsqueeze(0).cuda()
                outputs = self.forward(data)
                outputs = F.softmax(outputs)
                # outputs = F.sigmoid(outputs)
                preds.append(outputs.detach().cpu().numpy())
        preds = np.vstack(preds)

        df_crop = ds.df
        for i in range(preds.shape[-1]):
            df_crop[f"cat{i}"] = preds[:, i]

        dict_label = {}
        for image_id, gdf in df_crop.groupby("image_id"):
            dict_label[image_id] = np.argmax(
                gdf[[f"cat{i}" for i in range(preds.shape[-1])]].values.max(axis=0)
            )
            # dict_label[image_id] = np.argmax( gdf[ [f"cat{i}" for i in range(preds.shape[-1])] ].values.mean(axis=0) )
        preds = np.array(
            [dict_label[image_id] for image_id in df_crop["image_id"].unique()]
        )
        pred_labels = [ds.class_name[i] for i in preds]
        gt_df = ds.raw_df.copy()
        test_df = ds.raw_df.copy()
        test_df["label"] = pred_labels
        # calculate scores
        scores = score(
            solution=gt_df[["image_id", "label"]],
            submission=test_df[["image_id", "label"]],
            row_id_column_name="image_id",
        )
        return scores

    def on_validation_epoch_end(self):
        step = 10
        if self.current_epoch % step == 9:
            self.log("train/valid_scores", self.get_score(type="valid"))
            self.log("train/train_scores", self.get_score(type="train"))
        else:
            pass
