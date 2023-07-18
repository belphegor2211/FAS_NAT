import torch.nn as nn
from networks import get_model
from torch.utils.data import Dataset, DataLoader
from utils import *
from loss import supcon_loss

import time
import numpy as np
from torchvision import transforms, datasets
import argparse

from datasets.supcon_dataset import FaceDataset, DEVICE_INFOS

from datasets import get_datasets, TwoCropTransform

torch.backends.cudnn.benchmark = True



def parse_args():
    parser = argparse.ArgumentParser()
    # build dirs
    parser.add_argument('--data_dir', type=str, default="datasets/FAS", help='YOUR_Data_Dir')
    parser.add_argument('--result_path', type=str, default='./results', help='root result directory')
    # training settings
    parser.add_argument('--model_type', type=str, default="ResNet18_lgt", help='model_type')
    parser.add_argument('--img_size', type=int, default=256, help='img size')
    parser.add_argument('--pretrain', type=str, default='imagenet', help='imagenet')
    parser.add_argument('--batch_size', type=int, default=96, help='batch size')
    parser.add_argument('--normfc', type=str2bool, default=False)
    parser.add_argument('--usebias', type=str2bool, default=True)
    parser.add_argument('--feat_loss', type=str, default='supcon', help='')


    parser.add_argument('--device', type=str, default='0', help='device id, format is like 0,1,2')

    return parser.parse_args()


def str2bool(x):
    return x.lower() in ('true')


if __name__ == '__main__':
    args = parse_args()
    total_cls_num = 2
    inp = torch.rand(1,3,256,256)
    model = get_model(args.model_type, total_cls_num, pretrained=False, normed_fc=args.normfc, 
                      use_bias=args.usebias, simsiam=True if args.feat_loss == 'simsiam' else False)
    model.eval()
    _, _, logit = model(inp)
    scores_list = []
    print(logit.squeeze().item())
