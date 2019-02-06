# -*- coding:utf-8 -*-
__author__ = 'shichao'

import detection
import reid_api

import cv2
import numpy as np
import glob
import os
import sys
from random import randint


REID = reid_api.ReID(0, './weights/reID/PRID/60/ft_ResNet50')
det = detection.Detection(0, './cfg/yolov3.cfg', './cfg/yolov3.weights')
feature_set = []
previous_bbox = []
target_cam_ID = 0
CAM_SWITCH = False

def compute_iou(rec1, rec2):

    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])

    # computing the sum_area
    sum_area = S_rec1 + S_rec2


    left_line = max(rec1[1], rec2[1])
    right_line = min(rec1[3], rec2[3])
    top_line = max(rec1[0], rec2[0])
    bottom_line = min(rec1[2], rec2[2])


    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return float(intersect) / (sum_area - intersect)

def bbox_trick(last_bbox,new_bbox_set):
    thresh = 0.0
    index = -1
    for i in range(len(new_bbox_set)):
        rec1 = last_bbox
        rec2 = new_bbox_set[i]
        ratio = compute_iou(rec1,rec2)
        if ratio > thresh:
            index = i
    return index


def GetInitFeat(imgs,bbox,target_ID):
    '''
    get feature sets of initial frames
    :param imgs: some input 3-channel imgages
    :param bbox: one bbox selected by the user
    :param img_camIDs: the camIDs of input images, respectively
    :param target_camID: the camID of bbox
    :return: the feature sets of initial frames
    '''
    # global feature_set
    global REID
    global previous_bbox
    global target_cam_ID
    target_cam_ID = target_ID
    previous_bbox = bbox
    features = []
    # w,h,c,vol = imgs.shape
    vol = len(imgs)
    for i in range(vol):
        # frame = imgs[:,:,:,i]
        frame = imgs[i]
        init_feature,_ = REID.rank_feature(bbox, frame, features, init_frame=True)
        features.append(init_feature)
    return features


def FeedFeat(features):
    global feature_set
    feature_set = features

def GetBBox(imgs,img_camIDs):
    '''
    get the bbox of the target
    :param imgs: some 3-channel images
    :param img_camIDs: the camID of each image in the image gop
    :param frameIDs: the frameID of each image in the image gop
    :param target_camID: target in which camaras
    :return:
    a list of cam IDs
    a list of frame IDs
    a list of left top positions
    a list of right bottom positions
    '''
    global REID
    global det
    global previous_bbox
    global feature_set
    global CAM_SWITCH
    global target_cam_ID

    # w,h,c,vol = imgs.shape

    vol = len(imgs)

    # initialize p1,p2
    p1 = [30,35]
    p2 = [30,35]
    confidence_thresh = 0.6

    if not CAM_SWITCH:
        for i in range(vol):

            # get the target cam ID img
            if img_camIDs[i] == target_cam_ID:
                frame = imgs[i]
                w,h = frame.shape
                bbox_list = det.detect_one_frame(frame)
                match_feature_list, match_bbox_list, confidence_list = REID.rank_feature(bbox_list, frame, feature_set,
                                                                                         init_frame=False)
                # index = bbox_trick(previous_bbox, match_bbox_list)
                # if index < 0:
                #     index = confidence_list.index(max(confidence_list))

                index = confidence_list.index(max(confidence_list))

                if confidence_list[index] > 0.9:
                    match_feature = match_feature_list[index]
                    match_bbox = match_bbox_list[index]

                    feature_set.append(match_feature)
                    previous_bbox = match_bbox

                    p1 = (int(match_bbox[0]), int(match_bbox[1]))
                    p2 = (int(match_bbox[2]), int(match_bbox[3]))
                else:
                    p1 = (0,0)
                    p2 = (0,0)

            # search all camaras if the target is close to the boundary

            if min(p1) < 10 or max(p1) > (w - 10) or min(p2) < 0 or max(p2) > (h - 10):
                CAM_SWITCH = True

            return p1, p2, target_cam_ID


    else:
        for i in range(vol):
            frame = imgs[i]

            bbox_list = det.detect_one_frame(frame)
            match_feature_list, match_bbox_list, confidence_list = REID.rank_feature(bbox_list, frame,
                                                                                     feature_set,
                                                                                     init_frame=False)
            index = confidence_list.index(max(confidence_list))
            # index = bbox_trick(previous_bbox,match_bbox_list)
            # if index < 0:
            #     index = confidence_list.index(max(confidence_list))
            match_feature = match_feature_list[index]
            match_bbox = match_bbox_list[index]
            confidence = confidence_list[index]

            if confidence > confidence_thresh:
                w,h = frame.shape
                confidence_thresh = confidence
                feature_new = match_feature
                bbox_new = match_bbox
                target_cam_ID = img_camIDs[i]
                previous_bbox = match_bbox

        if bbox_new:
            p1 = (int(bbox_new[0]), int(bbox_new[1]))
            p2 = (int(bbox_new[2]), int(bbox_new[3]))
            previous_bbox = bbox_new
            feature_set.append(feature_new)

            if min(p1) < 10 or max(p1) > (w - 10) or min(p2) < 0 or max(p2) > (h - 10):
                CAM_SWITCH = True
            else:
                CAM_SWITCH = False

        else:
            p1 = []
            p2 = []
            target_cam_ID = 0
            CAM_SWITCH = True

        return p1, p2, target_cam_ID


def main():
    camID = 0
    # model_path = '/home/user/reID/demo/weights/reID/ft_ResNet50'  #marker

    global det
    global REID
    global feature_set
    global target_cam_ID

    # fetch the selected video
    video_init_path = "videos/cam_00_20190203.mp4"
    # Create a video capture object to read videos
    cap = cv2.VideoCapture(video_init_path)

    # fetch data from multi mp4 files into the video list
    video_all_paths = glob.glob(os.path.join('./videos', '*.mp4'))
    cap_list = []
    for i in range(len(video_all_paths)):
        cap_list.append(cv2.VideoCapture(os.path.join('videos', 'cam_' + str(i).zfill(2) + '_20190203.mp4')))

    cam_vol = len(cap_list)

    # Read first frame
    success, frame = cap.read()
    h, w, c = frame.shape

    # quit if unable to read the video file
    if not success:
        print('Failed to read video')
        sys.exit(1)

    ## Select boxes
    bbox_cv = cv2.selectROI('target ROI', frame)
    cv2.destroyAllWindows()
    bbox = list([[int(bbox_cv[0]), int(bbox_cv[1]), int(bbox_cv[0] + bbox_cv[2]), int(bbox_cv[1] + bbox_cv[3])], ])
    colors = (randint(64, 255), randint(64, 255), randint(64, 255))


    init_img_set = []
    init_img_set.append(frame)
    while i < 5:
        i += 1
        success, frame = cap.read()
        if not success:
            break
        init_img_set.append(frame)

    feat_sets = GetInitFeat(init_img_set,bbox,camID)
    FeedFeat(feat_sets)

    img_set = []
    counter = 0
    while cap.isOpened():
        if not success:
            break
        for i in range(cam_vol):
            cap = cap_list[i]
            success, frame = cap.read()
            img_set.append(frame)

        p1,p2,target_cam_ID = GetBBox(img_set,target_cam_ID)
        img = img_set[target_cam_ID]
        cv2.rectangle(img, p1, p2, colors[0], 2, 1)
        cv2.imwrite(os.path.join('./results', 'c{0}'.format(target_cam_ID)+'f'+str(counter).zfill(5) + '.jpg'), img)
        counter += 1


if __name__ == '__main__':
    main()
