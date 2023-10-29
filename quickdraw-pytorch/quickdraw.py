#!/usr/bin/env python
# coding: utf-8

# Install required packages manually:
# pip3 install transformers wandb

import wandb

# <a href="https://colab.research.google.com/github/nateraw/quickdraw-pytorch/blob/main/quickdraw.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

wandb.login()

from typing import List, Optional
import urllib.request
from tqdm.auto import tqdm
from pathlib import Path
import requests
import torch
import math
import numpy as np
import os
import glob


def get_quickdraw_class_names():
    """
    TODO - Check performance w/ gsutil in colab. The following command downloads all files to ./data
    `gsutil cp gs://quickdraw_dataset/full/numpy_bitmap/* ./data`
    """
    url = "https://raw.githubusercontent.com/googlecreativelab/quickdraw-dataset/master/categories.txt"
    r = requests.get(url)
    classes = [x.replace(' ', '_') for x in r.text.splitlines()]
    return classes


def download_quickdraw_dataset(root="./data", limit: Optional[int] = None, class_names: List[str]=None):
    if class_names is None:
        class_names = get_quickdraw_class_names()

    root = Path(root)
    root.mkdir(exist_ok=True, parents=True)
    url = 'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/'

    print("Downloading Quickdraw Dataset...")
    for class_name in tqdm(class_names[:limit]):
        fpath = root / f"{class_name}.npy"
        if not fpath.exists():
            urllib.request.urlretrieve(f"{url}{class_name.replace('_', '%20')}.npy", fpath)


def load_quickdraw_data(root="./data", max_items_per_class=500):
    all_files = Path(root).glob('*.npy')

    x = np.empty([0, 784], dtype=np.uint8)
    y = np.empty([0], dtype=np.int64)
    class_names = []

    print(f"Loading {max_items_per_class} examples for each class from the Quickdraw Dataset...")
    for idx, file in enumerate(tqdm(sorted(all_files))):
        data = np.load(file, mmap_mode='r')
        data = data[0: max_items_per_class, :]
        labels = np.full(data.shape[0], idx)
        x = np.concatenate((x, data), axis=0)
        y = np.append(y, labels)

        class_names.append(file.stem)

    return x, y, class_names


class QuickDrawDataset(torch.utils.data.Dataset):
    def __init__(self, root, max_items_per_class=5000, class_limit=None):
        super().__init__()
        self.root = root
        self.max_items_per_class = max_items_per_class
        self.class_limit = class_limit

        download_quickdraw_dataset(self.root, self.class_limit)
        self.X, self.Y, self.classes = load_quickdraw_data(self.root, self.max_items_per_class)

    def __getitem__(self, idx):
        x = (self.X[idx] / 255.).astype(np.float32).reshape(1, 28, 28)
        y = self.Y[idx]

        return torch.from_numpy(x), y.item()

    def __len__(self):
        return len(self.X)

    def collate_fn(self, batch):
        x = torch.stack([item[0] for item in batch])
        y = torch.LongTensor([item[1] for item in batch])
        return {'pixel_values': x, 'labels': y}
    
    def split(self, pct=0.1):
        num_classes = len(self.classes)
        indices = torch.randperm(len(self)).tolist()
        n_val = math.floor(len(indices) * pct)
        train_ds = torch.utils.data.Subset(self, indices[:-n_val])
        val_ds = torch.utils.data.Subset(self, indices[-n_val:])
        return train_ds, val_ds


import torch
from transformers import Trainer
from transformers.modeling_utils import ModelOutput


class QuickDrawTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False):
        logits = model(inputs["pixel_values"])
        labels = inputs.get("labels")

        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return (loss, ModelOutput(logits=logits, loss=loss)) if return_outputs else loss

# Taken from timm - https://github.com/rwightman/pytorch-image-models/blob/master/timm/utils/metrics.py
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = min(max(topk), output.size()[1])
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:min(k, maxk)].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]


def quickdraw_compute_metrics(p):
    acc1, acc5 = accuracy(
        torch.tensor(p.predictions),
        torch.tensor(p.label_ids), topk=(1, 5)
    )
    return {'acc1': acc1, 'acc5': acc5}


# In[ ]:


import torch
from torch import nn
from transformers import TrainingArguments
from datetime import datetime

data_dir = './data'
max_examples_per_class = 1000
train_val_split_pct = .1

ds = QuickDrawDataset(data_dir, max_examples_per_class)
num_classes = len(ds.classes)
train_ds, val_ds = ds.split(train_val_split_pct)


# ## Define the Model

# In[ ]:


# Original model used for hf.co/spaces/nateraw/quickdraw
# When I trained the model for the spaces demo (link above), I limited to 100 classes. Full dataset is >300 classes
# I'm updating to use the model uncommented below, as its slightly bigger and works better on the full dataset. 
# model = nn.Sequential(
#     nn.Conv2d(1, 32, 3, padding='same'),
#     nn.ReLU(),
#     nn.MaxPool2d(2),
#     nn.Conv2d(32, 64, 3, padding='same'),
#     nn.ReLU(),
#     nn.MaxPool2d(2),
#     nn.Conv2d(64, 128, 3, padding='same'),
#     nn.ReLU(),
#     nn.MaxPool2d(2),
#     nn.Flatten(),
#     nn.Linear(1152, 512),
#     nn.ReLU(),
#     nn.Linear(512, num_classes),  # num_classes was limited to 100 here
# )

model = nn.Sequential(
    nn.Conv2d(1, 64, 3, padding='same'),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(64, 128, 3, padding='same'),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(128, 256, 3, padding='same'),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(2304, 512),
    nn.ReLU(),
    nn.Linear(512, num_classes),
)


# Train



timestamp = datetime.now().strftime('%Y-%m-%d-%H%M%S')
training_args = TrainingArguments(
    output_dir=f'./outputs_20k_{timestamp}',
    evaluation_strategy='epoch',
    save_strategy='epoch',
    report_to=['wandb', 'tensorboard'],  # Update to just tensorboard if not using wandb
    logging_strategy='steps',
    logging_steps=100,
    per_device_train_batch_size=256,
    per_device_eval_batch_size=256,
    learning_rate=0.003,
    fp16=torch.cuda.is_available(),
    num_train_epochs=20,
    run_name=f"quickdraw-med-{timestamp}",  # Can remove if not using wandb
    warmup_steps=10000,
    save_total_limit=5,
)

trainer = QuickDrawTrainer(
    model,
    training_args,
    data_collator=ds.collate_fn,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=None,
    compute_metrics=quickdraw_compute_metrics,
)

try:
    # Training
    train_results = trainer.train()

    trainer.save_model()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()

    # Evaluation
    eval_results = trainer.evaluate()
    trainer.log_metrics("eval", eval_results)
    trainer.save_metrics("eval", eval_results)
except:
    pass
finally:
    # Save the model's weights to 'pytorch_model.bin'
    torch.save(model.state_dict(), 'pytorch_model.bin')
    wandb.finish()