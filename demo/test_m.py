# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import scipy.io
from model import ft_net, ft_net_dense, PCB, PCB_test
from PIL import Image

import torch.nn as nn
from torchvision import datasets, models, transforms
import os
from model import ft_net, ft_net_dense, PCB, PCB_test
import utils

import argparse
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
import time
import scipy.io
from reid_api import ReID

########################################################################


class ReID:
    def __init__(self, gpuID, model_path):
        ####################################################
        # Options
        # ------------------
        self.gpu_ids = 1
        self.which_epoch = '59'
        self.batch_size = 1
        self.use_dense = False
        self.use_PCB = False
        self.model_path = model_path
        self.class_num = 751
        self.score_threshold = 0.9
        self.confidence_threshold = 0.6
        ####################################################
        # Set gpu
        # ------------------
        use_gpu = torch.cuda.is_available()
        if not use_gpu:
            print('can not user gpu')
            exit()
        torch.cuda.set_device(self.gpu_ids)

        ####################################################
        # Load model
        # ------------------
        print('load model...')
        if self.use_dense:
            model_structure = ft_net_dense(self.class_num)
        else:
            model_structure = ft_net(self.class_num)
        if self.use_PCB:
            model_structure = PCB(self.class_num)

        model = utils.load_network(model_structure, self.model_path, self.which_epoch)
        # Remove the final fc layer and classifier layer
        if not self.use_PCB:
            model.model.fc = nn.Sequential()
            model.classifier = nn.Sequential()
        else:
            model = PCB_test(model)
        model = model.eval()
        #print(model)
        if use_gpu:
            model = model.cuda()
        self.model = model
        ####################################################
        # Set Transform
        # ------------------
        print('set transform...')
        self.data_transforms = transforms.Compose([
            transforms.Resize((256, 128), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        if self.use_PCB:
            self.data_transforms = transforms.Compose([
                transforms.Resize((384, 192), interpolation=3),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def rank_feature(self, box_list, image, feature_set, init_frame=False):
        #################################################################
        # Load data
        # -----------------------------------
        print('load data...')
        #feature_set = transforms.ToTensor(feature_set)
        crop_list = []
        bbox_list = list(bbox_list)
        print(type(bbox_list))
        print(bbox_list)
        for k in range(len(bbox_list)):
            #print(bbox_list[k][0])
            if bbox_list[k][0] < 0:
                bbox_list[k][0] = 0
            if bbox_list[k][1] < 0:
                bbox_list[k][1] = 0
            if bbox_list[k][2] < 0:
                bbox_list[k][2] = 0
            if bbox_list[k][3] < 0:
                bbox_list[k][3] = 0
            box = (bbox_list[k][0], bbox_list[k][1], bbox_list[k][2], bbox_list[k][3])
            cropped = image.crop(box)

            crop_list.append(cropped)

        data_loaders = torch.utils.data.DataLoader(self.ReidDataset(crop_list, self.data_transforms),
                                                   batch_size=self.batch_size,
                                                   shuffle=False, sampler=None, num_workers=16)
        ######################################################################
        # get feature
        # ---------------------------
        print('get feature...')
        with torch.no_grad():
            query_feature = utils.extract_feature(self.model, data_loaders, self.use_dense, self.use_PCB)

        #####################################################################
        # get list
        # ---------------------------
        print('get list...')
        if init_frame:
            return query_feature, bbox_list, _
        else:
            #feature_set = Variable(feature_set.cuda())
            feature_set = feature_set.cuda()
            feature_list = []
            box_list = []
            confidence_list = []
            for i in range(len(query_feature)):
                confidence, box_num = utils.evaluate(query_feature[i], feature_set, i,
                                                     self.score_threshold, self.confidence_threshold)
                feature_list = feature_list.append(query_feature[box_num])
                box_list = box_list.append(bbox_list[box_num])
                confidence_list = confidence_list.append(confidence)
            return feature_list, box_list, confidence_list

def loadDatadet(infile,k):
    f=open(infile,'r')
    sourceInLine=f.readlines()
    dataset=[]
    for line in sourceInLine:
        temp1=line.strip('\n')
        temp2=temp1.split(' ')
        dataset.append(temp2)
    for i in range(0,len(dataset)):
        for j in range(k):
            dataset[i].append(int(dataset[i][j]))
        del(dataset[i][0:k])
    return dataset

def main():
    infile = '/home/data/tongfang/lyj/raw_IP2/annotation_77.txt'
    k = 4
    bbox_list = loadDatadet(infile, k)
    print(bbox_list)
    file_path = '/home/data/tongfang/lyj/raw_IP2/raw00077.jpg'
    im = Image.open(file_path)
    model_path = '/home/user/lyj_ReID/tyrant_0110/Person_reID_baseline_pytorch-master/model/ft_ResNet50'
    REID = ReID('0', model_path)

    #feature_set = np.array(1).reshape(50, 2048)
    #m,n = map(int, input().split())
    #feature_set = [[0]*(m)]*(n)
    feature_sets = list(np.zeros((1,2048)))
    feature_sets.append(feature_sets)
    feature_set = feature_sets[0]
    #print(feature_set)
    feature_list, box_list, confidence_list = REID.rank_feature(bbox_list, im, feature_set, init_frame=False)
    #print(feature_list)
    #print(box_list)
    #print(confidence_list)


if __name__ == "__main__":
    main()


