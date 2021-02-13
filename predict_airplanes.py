# --------------------------------------------------------
# Written by Hamadi Chihaoui at 9:56 PM 2/13/2021 
# --------------------------------------------------------
import os
import numpy as np
import pandas as pd
from datetime import datetime
import time
import random
from tqdm.autonotebook import tqdm

# Torch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler

# sklearn
from sklearn.model_selection import StratifiedKFold

# CV
import cv2

################# DETR FUCNTIONS FOR LOSS########################
import sys

sys.path.append('./detr/')

from detr.models.matcher import HungarianMatcher
from detr.models.detr import SetCriterion
#################################################################

# Albumenatations
import albumentations as A
import matplotlib.pyplot as plt
from albumentations.pytorch.transforms import ToTensorV2

# Glob
from glob import glob

num_classes = 2
num_queries = 100


class DETRModel(nn.Module):
    def __init__(self, num_classes, num_queries):
        super(DETRModel, self).__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries

        self.model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
        self.in_features = self.model.class_embed.in_features

        self.model.class_embed = nn.Linear(in_features=self.in_features, out_features=self.num_classes)
        self.model.num_queries = self.num_queries

    def forward(self, images):
        return self.model(images)


model = DETRModel(num_classes=num_classes, num_queries=num_queries)
model.load_state_dict(torch.load('detr_best_0.pth'))
model = model.cuda()
model.eval()
img = cv2.imread('/home/hamadic/valid/images/93_104001000493AA00_tile_7.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
img1 = img.copy()
img /= 255.0
h, w, _ = img.shape
image = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)
print('image', image.shape, 'h', h, 'w', w)
with torch.no_grad():
    outputs = model(image.cuda())

    outputs = [{k: v.to('cpu') for k, v in outputs.items()}]

    oboxes = outputs[0]['pred_boxes'][0].detach().cpu().numpy()
    oboxes = [np.array(box).astype(np.int32) for box in A.augmentations.bbox_utils.denormalize_bboxes(oboxes, h, w)]
    prob = outputs[0]['pred_logits'][0].softmax(1).detach().cpu().numpy()[:, 0]

    for box, p in zip(oboxes, prob):

        if p > 0.5:
            color = (0, 0, 220)  # if p>0.5 else (0,0,0)
            img = cv2.rectangle(img1,
                                (box[0], box[1]),
                                (box[2] + box[0], box[3] + box[1]),
                                color, 2)
            cv2.imwrite('img.jpg', img)

