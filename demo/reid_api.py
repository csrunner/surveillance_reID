# -*- coding: utf-8 -*-

from __future__ import print_function, division

import torch
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
from PIL import Image
########################################################################


class ReID:
    def __init__(self, gpuID, model_path):
        ####################################################
        # Options
        # ------------------
        self.gpu_ids = gpuID
        self.which_epoch = 'last'
        self.batch_size = 1
        self.use_dense = False
        self.use_PCB = False
        self.model_path = model_path
        self.class_num = 749
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

        #save_path = os.path.join(self.model_path, 'net_%s.pth' % self.which_epoch)
        #print(save_path)
        #model = model_structure.load_state_dict(torch.load(save_path))
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

        self.feature_transforms = transforms.Compose([
            transforms.ToTensor()
            ])


    def rank_feature(self, bbox_list, img, feature_set, init_frame=False):
        #################################################################
        # Load data
        # -----------------------------------
        print('load data...')
        #feature_set = transforms.ToTensor(feature_set)
        crop_list = []
        image = Image.fromarray(img, mode='RGB')
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
            # cropped = image[bbox_list[bbox][1]:bbox_list[bbox][3], bbox_list[bbox][0]:bbox_list[bbox][2]]
            crop_list.append(cropped)

        data_loaders = torch.utils.data.DataLoader(utils.ReidDataset(crop_list, self.data_transforms),
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
            return query_feature[0], bbox_list
        else:
            #feature_set = Variable(feature_set.cuda())
            #gallery = torch.utils.data.DataLoader(utils.ReidDataset(feature_set, self.feature_transforms),
            #                                           batch_size=self.batch_size,
            #                                           shuffle=False, sampler=None, num_workers=16)
            gallery_set = []
            gallery_set = torch.FloatTensor()


            #feature_set = torch.utils.data.DataLoader(utils.ReidDataset(gallery_set, self.feature_transforms),
            #                                           batch_size=self.batch_size,
            #                                           shuffle=False, sampler=None, num_workers=16)

            print('feature_set.shape')
            print([len(a) for a in feature_set])
            for i in feature_set:
                im = i
                #im = transforms.ToTensor(i)
                #print('im==========')
                #print(im.shape)
                #imm = im
                im = im.view(1, 2048)
                #print('im+++++++++++')
                #print(im.shape)
                gallery_set = torch.cat((gallery_set, im), 0)

            gallery_set = gallery_set.cuda()
            query_feature_g = query_feature.cuda()
            #for i in gallery:
            #    im = i
            #    im = Variable(im.cuda())
            #    gallery_set = torch.cat((gallery_set,im), 0)
            feature_list = []
            box_list = []
            confidence_list = []

            #print('qf---------')
            #print(query_feature_g.shape)
            #print('qf1111111111')
            #print(query_feature_g[0].shape)
            #print('gf---------')
            #print(gallery_set.shape)
            
            for i in range(len(query_feature)):
                confidence, box_num, con = utils.evaluate(query_feature_g[i], gallery_set, i,
                                                          self.score_threshold, self.confidence_threshold)
                if con:
                    continue 
                
                print(query_feature[box_num])
                feature_list.append(query_feature[box_num])
                box_list.append(bbox_list[box_num])
                confidence_list.append(confidence)
            return feature_list, box_list, confidence_list













