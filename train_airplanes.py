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


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


n_folds = 5
seed = 42
num_classes = 2
num_queries = 100
null_class_coef = 0.5
BATCH_SIZE = 8
LR = 2e-5
EPOCHS = 2


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


seed_everything(seed)


def get_train_transforms():
    return A.Compose(
        [A.OneOf([A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.9),

                  A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.9)], p=0.9),

         A.ToGray(p=0.01),

         A.HorizontalFlip(p=0.5),

         A.VerticalFlip(p=0.5),

         A.Resize(height=512, width=512, p=1),

         A.Cutout(num_holes=8, max_h_size=64, max_w_size=64, fill_value=0, p=0.5),

         ToTensorV2(p=1.0)],

        p=1.0,

        bbox_params=A.BboxParams(format='coco', min_area=0, min_visibility=0, label_fields=['labels'])
        )


def get_valid_transforms():
    return A.Compose([A.Resize(height=512, width=512, p=1.0),
                      ToTensorV2(p=1.0)],
                     p=1.0,
                     bbox_params=A.BboxParams(format='coco', min_area=0, min_visibility=0, label_fields=['labels'])
                     )


DIR_TRAIN = '../input/global-wheat-detection/train'


class WheatDataset(Dataset):
    def __init__(self, path, transforms=None):
        self.path = path
        self.image_paths = os.listdir(self.path + 'images/')

        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.image_paths)

    def get_boxes(self, image_name):
        with open(self.path + 'labels/' + image_name[:-4] + '.txt', 'r') as f:
            data = f.read().split('\n')[:-1]
            boxes = []
            for d in data:
                splits = d.split(' ')
                box = [max(0., float(splits[1]) - float(splits[3]) / 2),
                       max(0., float(splits[2]) - float(splits[4]) / 2), float(splits[3]), float(splits[4])]
                boxes.append(box)
        return np.array(boxes)

    def __getitem__(self, index):
        image_name = self.image_paths[index]

        # print('image', self.path + 'images/' + image_name)
        image = cv2.imread(self.path + 'images/' + image_name, cv2.IMREAD_COLOR)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        h, w, _ = image.shape

        # DETR takes in data in coco format

        boxes = self.get_boxes(image_name) * np.array([w, h, w * 0.99, h * 0.99])

        # Area of bb
        area = boxes[:, 2] * boxes[:, 3]
        area = torch.as_tensor(area, dtype=torch.float32)

        # AS pointed out by PRVI It works better if the main class is labelled as zero
        labels = np.zeros(len(boxes), dtype=np.int32)

        if self.transforms:
            sample = {
                'image': image,
                'bboxes': boxes,
                'labels': labels
            }
            sample = self.transforms(**sample)
            image = sample['image']
            boxes = sample['bboxes']
            labels = sample['labels']

        # Normalizing BBOXES

        _, h, w = image.shape
        boxes = A.augmentations.bbox_utils.normalize_bboxes(sample['bboxes'], rows=h, cols=w)
        target = {}
        target['boxes'] = torch.as_tensor(boxes, dtype=torch.float32)
        target['labels'] = torch.as_tensor(labels, dtype=torch.long)
        target['image_id'] = torch.tensor([index])
        target['area'] = area

        return image, target, image_name


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


'''
code taken from github repo detr , 'code present in engine.py'
'''

matcher = HungarianMatcher()

weight_dict = weight_dict = {'loss_ce': 1, 'loss_bbox': 1, 'loss_giou': 1}

losses = ['labels', 'boxes', 'cardinality']


def iou(bboxes_preds, bboxes_targets):
    area1 = (bboxes_preds[:, 2] - bboxes_preds[:, 0]) * (bboxes_preds[:, 3] - bboxes_preds[:, 1])
    area2 = (bboxes_targets[:, 2] - bboxes_targets[:, 0]) * (bboxes_targets[:, 3] - bboxes_targets[:, 1])
    width = (torch.min(bboxes_preds[:, 2, None], bboxes_targets[:, 2]) -
             torch.max(bboxes_preds[:, 0, None], bboxes_targets[:, 0])).clamp(min=0)
    height = (torch.min(bboxes_preds[:, 3, None], bboxes_targets[:, 3]) -
              torch.max(bboxes_preds[:, 1, None], bboxes_targets[:, 1])).clamp(min=0)
    inter = width * height
    return inter / (area1[:, None] + area2 - inter)  # p * t


def average_precision(bbox_preds, conf_preds, bbox_targets):
    if len(bbox_targets) == 0 and len(bbox_preds) == 0:
        # print("no predictions and no ground truth")
        return 1.0, 1.0, 1.0
    elif len(bbox_targets) == 0:
        # print("no ground truth")
        return 0.0, 1.0, 0.0
    elif len(bbox_preds) == 0:
        # print("no predictions")
        return 0.0, 0.0, 1.0
    else:
        thresholds = [0.5]  # 0.55, 0.6, 0.65, 0.7, 0.75
        bbox_preds = bbox_preds[torch.argsort(conf_preds, descending=True)]
        scores = iou(bbox_preds, bbox_targets)
        precision = 0.0
        rec = 0.0
        pr = 0.0
        for threshold in thresholds:
            matched_targets = torch.zeros(bbox_targets.size(0))
            tp = 0
            fp = 0
            fn = 0
            for score in scores:
                score[matched_targets == 1] = 0.0
                s, i = score.max(dim=0)
                if s >= threshold:
                    tp += 1
                    matched_targets[i] = 1
                else:
                    fp += 1
            fn = torch.sum((matched_targets == 0)).item()
            precision += tp / (tp + fp + fn)
            rec += tp / (tp + fn)
            pr += tp / (tp + fp)
        precision /= len(thresholds)
        rec /= len(thresholds)
        pr /= len(thresholds)
        return precision, rec, pr


def calculate_map(dets, scores, targets, batch_size, device):
    sc_precision = 0.0
    sc_rec = 0.0
    sc_pr = 0.0
    score_threshold = 0.5
    for i in range(batch_size):
        try:
            boxes = dets[i][:, :4]
            print('boxes', boxes)
            print('scores', scores)
        except:
            print('dets[i]', dets[i])
        # scores = dets[i][:,4]
        # indexes = torch.where(scores > score_threshold)[0]
        # boxes = boxes[indexes]
        # scores = scores[indexes]
        target_boxes = targets[i]['boxes']
        # target_boxes[:,[0,1,2,3]] = target_boxes[:,[1,0,3,2]]
        # boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        # boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        av_pre, rec, pr = average_precision(boxes, scores, target_boxes.to(device))
        sc_precision = sc_precision + av_pre
        sc_rec = sc_rec + rec
        sc_pr = sc_pr + pr
    return sc_precision, sc_rec, sc_pr


def train_fn(data_loader, model, criterion, optimizer, device, scheduler, epoch):
    model.train()
    criterion.train()

    summary_loss = AverageMeter()

    tk0 = tqdm(data_loader, total=len(data_loader))

    for step, (images, targets, image_ids) in enumerate(tk0):

        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        output = model(images)

        loss_dict = criterion(output, targets)
        weight_dict = criterion.weight_dict

        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        optimizer.zero_grad()

        losses.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        summary_loss.update(losses.item(), BATCH_SIZE)
        tk0.set_postfix(loss=summary_loss.avg)

    return summary_loss


def eval_fn(data_loader, model, criterion, device):
    model.eval()
    criterion.eval()
    summary_loss = AverageMeter()
    cumm_pr = 0.0
    cumm_rec = 0.0
    len_viewed = 0.0
    m = nn.Softmax(dim=-1)

    with torch.no_grad():

        tk0 = tqdm(data_loader, total=len(data_loader))
        for step, (images, targets, image_ids) in enumerate(tk0):

            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            output = model(images)
            detections = output['pred_boxes']
            logits = output['pred_logits']
            probs = m(logits)
            final_detections = []
            final_scores = []
            for i in range(len(images)):
                probs_i = probs[i, :, 0]
                indices = torch.where(probs_i > 0.5)
                scores = probs_i[indices]
                filtered = detections[i][indices] * 1024
                filtered[:, 2] = filtered[:, 2] + filtered[:, 0]
                filtered[:, 3] = filtered[:, 3] + filtered[:, 1]
                final_detections.append(filtered)
                final_scores.append(scores)

            sc_precision, sc_rec, sc_pr = calculate_map(final_detections, final_scores, targets, len(images),
                                                        torch.device('cuda:0'))
            len_viewed = len_viewed + len(images)
            cumm_pr = cumm_pr + sc_pr
            cumm_rec = cumm_rec + sc_rec

            loss_dict = criterion(output, targets)
            weight_dict = criterion.weight_dict

            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

            summary_loss.update(losses.item(), BATCH_SIZE)
            tk0.set_postfix(loss=summary_loss.avg)

    return summary_loss, cumm_pr / len_viewed, cumm_rec / len_viewed


def collate_fn(batch):
    return tuple(zip(*batch))


def run(fold):
    train_dataset = WheatDataset(
        '/home/hamadic/Kag/global_wheat_detection/yolov5/train/',
        transforms=get_train_transforms()
    )

    valid_dataset = WheatDataset(
        '/home/hamadic/Kag/global_wheat_detection/yolov5/valid/',
        transforms=get_valid_transforms()
    )

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )

    valid_data_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )

    device = torch.device('cuda')
    model = DETRModel(num_classes=num_classes, num_queries=num_queries)
    model = model.to(device)
    criterion = SetCriterion(num_classes - 1, matcher, weight_dict, eos_coef=null_class_coef, losses=losses)
    criterion = criterion.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    best_loss = 10 ** 5
    for epoch in range(EPOCHS):
        train_loss = train_fn(train_data_loader, model, criterion, optimizer, device, scheduler=None, epoch=epoch)
        valid_loss, cumm_pr, cumm_rec = eval_fn(valid_data_loader, model, criterion, device)

        print('|EPOCH {}| TRAIN_LOSS {}| VALID_LOSS {}|  Prec {}|  Recall {}'.format(epoch + 1, train_loss.avg,
                                                                                     valid_loss.avg, cumm_pr, cumm_rec))

        if valid_loss.avg < best_loss:
            best_loss = valid_loss.avg
            print('Best model found for Fold {} in Epoch {}........Saving Model'.format(fold, epoch + 1))
            torch.save(model.state_dict(), f'detr_best_{fold}.pth')


run(fold=0)
