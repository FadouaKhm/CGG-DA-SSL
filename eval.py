# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 14:08:12 2022

@author: Fadoua Khmaissia
"""
from __future__ import print_function, division
import os
from config import config

import torch
from sklearn.metrics import *
import numpy as np
from utils.utils import net_builder
from datasets.ssl_dataset import SSL_Dataset
from datasets.data_utils import get_data_loader
from utils.plot_cf import *
from sklearn.manifold import TSNE
from collections import Counter
import os
import argparse
from sklearn.metrics import classification_report
import numpy as np
import seaborn as sns
import pandas as pd

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--load_path', type=str, default='./saved_models/fs_3500/fs3.pth') 
    parser.add_argument('--use_train_model', action='store_true')

    '''
    Backbone Net Configurations
    '''
    parser.add_argument('--net', type=str, default='vgg16_bn')#WideResNetVar
    parser.add_argument('--net_from_name', type=bool, default=True)
    parser.add_argument('--depth', type=int, default=28)
    parser.add_argument('--widen_factor', type=int, default=2)
    parser.add_argument('--leaky_slope', type=float, default=0.1)
    parser.add_argument('--dropout', type=float, default=0.0)

    '''
    Data Configurations
    '''
    parser.add_argument('--test', default="./inputs/test.txt" , help="Path to the list of validation videos")
    parser.add_argument('--data_files_dir', default= r'C:\Users\user\Desktop\TRAIN_SSL - 5\data\saved_data_8cls' , help="Path extracted and preprocessed ATR videos videos")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--data_dir', type=str, default='')
    parser.add_argument('--dataset', type=str, default='ATR')
    parser.add_argument('--num_classes', type=int, default=2)
    args = parser.parse_args()
    
    checkpoint_path = os.path.join(args.load_path)
    checkpoint = torch.load(checkpoint_path)
    load_model = checkpoint['ema_model']
    args.bn_momentum = 1.0 - 0.999

    _net_builder = net_builder(args.net, 
                                args.net_from_name,
                                {'depth': args.depth, 
                                'widen_factor': args.widen_factor,
                                'leaky_slope': args.leaky_slope,
                                'dropRate': args.dropout,
                                'use_embed': False})
    
    net = _net_builder(num_classes=args.num_classes)
    net.load_state_dict(load_model)
    #net = torch.nn.DataParallel(net).cuda()

    if torch.cuda.is_available():
        net.cuda()
    #net.load_state_dict(load_model)
    net.eval()
    test_files = open(args.test, "r").read().splitlines()
    
    _eval_dset = SSL_Dataset(args, alg='fullysupervised', name=args.dataset, train=False,
                              num_classes=args.num_classes, data_dir=args.data_dir)
    eval_dset = _eval_dset.get_dset(txt_data_dir=args.data_files_dir, lb_files= test_files)
    
    eval_loader = get_data_loader(eval_dset,
                                  args.batch_size, 
                                  num_workers=1, drop_last=False)
 
    acc = 0.0
    y_true = []
    y_pred = []
    y_logits = []
    feats = []
    with torch.no_grad():
        for _, image, target in eval_loader:

            image = image.type(torch.FloatTensor).cuda()
            logit = net(image)
            

            
            y_true.extend(target.cpu().tolist())
            y_pred.extend(torch.max(logit, dim=-1)[1].cpu().tolist()) 
            y_logits.extend(torch.softmax(logit, dim=-1).cpu().tolist())
            
            acc += logit.cpu().max(1)[1].eq(target).sum().numpy()
    print('Test accuracy = ', accuracy_score(y_true, y_pred))

