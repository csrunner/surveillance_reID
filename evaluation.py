# -*- coding:utf-8 -*-
__author__ = 'shichao'
import numpy as np
np.argsort()
def compute_AP(good_image,junk_image,index):
    cmc = np.zeros(len(index),1)
    ngood = len(good_image)
    old_recall = 0
    old_precision = 1.0
    ap = 0
    intersect_size = 0
    j = 0
    good_now = 0
    njunk = 0
    for n in range(len(index)):
        flag = 0
        good_image_index = np.where(good_image==index[n])
        if good_image_index:
            flag = 1
            good_now += 1
        junk_image_index = np.where(junk_image==index[n])
        if junk_image_index:
            njunk += 1
            continue

        if flag == 1:
            intersect_size += 1
        recall = intersect_size/float(ngood)
        precision = intersect_size/float(j+1)
        ap += (recall-old_recall)*((old_precision+precision)/2)
        old_recall = recall
        old_precision = precision
        j += 1

        if good_now == ngood:
            break
    return ap

def main():
    good_image = ''
    junk_image = ''
    index = ''
    ap =
