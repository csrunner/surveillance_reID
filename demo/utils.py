# -*- coding: utf-8 -*-

from __future__ import print_function, division

import torch
from torch.autograd import Variable
import numpy as np
import os

class ReidDataset(torch.utils.data.Dataset):
    def __init__(self, data_list, transform=None):
        self.imgs = data_list
        self.transform = transform
    def __getitem__(self, index):
        img = self.imgs[index]
        if self.transform is not None:
            img = self.transform(img)
        return img
    def __len__(self):
        return len(self.imgs)

def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def extract_feature(model, dataloaders, use_dense=False, use_PCB=False):
    features = torch.FloatTensor()
    count = 0
    for data in dataloaders:
        img = data
        #print('img size:')
        #print(img.size())
        n, c, h, w = img.size()
        count += n
        print(count)
        if use_dense:
            ff = torch.FloatTensor(n,1024).zero_()
        else:
            ff = torch.FloatTensor(n,2048).zero_()                #512
        if use_PCB:
            ff = torch.FloatTensor(n,2048,6).zero_() # we have six parts
        for i in range(2):
            if(i==1):
                img = fliplr(img)
            input_img = Variable(img.cuda())
            outputs = model(input_img)
            f = outputs.data.cpu()
            ff = ff+f
        # norm feature
        if use_PCB:
            # feature size (n,2048,6)
            # 1. To treat every part equally, I calculate the norm for every 2048-dim part feature.
            # 2. To keep the cosine score==1, sqrt(6) is added to norm the whole feature (2048*6).
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(6)
            ff = ff.div(fnorm.expand_as(ff))
            ff = ff.view(ff.size(0), -1)
        else:
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))

        features = torch.cat((features,ff), 0)
    return features


def evaluate(qf, gf, i, score_threshold, confidence_threshold):
    query = qf.view(-1, 1)
    score = torch.mm(gf, query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    score_len = len(score)
    good=0
    for each_score in score:
        if each_score >= score_threshold:
            good += 1
    confidence = float(good / score_len)
    con = False
    if confidence <= confidence_threshold:
        con = True
        return confidence, i, con
    else:
        good_confidence = confidence
        return good_confidence, i, con

def load_network(network, model_path, which_epoch):
    save_path = os.path.join(model_path, 'net_%s.pth' % which_epoch)
    print(save_path)
    #save_path = os.path.join('./model',name,'net_%s.pth'%opt.which_epoch)
    network.load_state_dict(torch.load(save_path))
    return network















