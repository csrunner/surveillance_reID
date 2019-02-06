# -*- coding:utf-8 -*-
__author__ = 'shichao'


import cv2
import os
import numpy
import parser
import sys
from random import randint
import glob

import detection
import reid_api


def compute_iou(rec1, rec2):
    """
    computing IoU
    :param rec1: (x0, y0, x1, y1), which reflects
            (top, left, bottom, right)
    :param rec2: (x0, y0, x1, y1)
    :return: scala value of IoU
    """
    # computing area of each rectangles
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])

    # computing the sum_area
    sum_area = S_rec1 + S_rec2

    # find the each edge of intersect rectangle
    left_line = max(rec1[1], rec2[1])
    right_line = min(rec1[3], rec2[3])
    top_line = max(rec1[0], rec2[0])
    bottom_line = min(rec1[2], rec2[2])

    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return float(intersect) / (sum_area - intersect)

def bbox_trick(last_bbox,new_bbox_set):
    thresh = 0.1
    index = -1
    for i in range(len(new_bbox_set)):
        rec1 = last_bbox
        rec2 = new_bbox_set[i]
        ratio = compute_iou(rec1,rec2)
        if ratio > thresh:
            index = i
    return index


def main():
    model_path = './weights/reID/PRID/60/ft_ResNet50'  # person-2011

    REID = reid_api.ReID(0, model_path)

    det = detection.Detection(0, './cfg/yolov3.cfg', './cfg/yolov3.weights')

    # fetch the selected video
    video_init_path = "videos/cam_00_20190203.mp4"
    # Create a video capture object to read videos
    cap = cv2.VideoCapture(video_init_path)

    # fetch data from multi mp4 files into the video list
    video_all_paths = glob.glob(os.path.join('./videos', '*.mp4'))
    cap_list = []
    for i in range(len(video_all_paths)):
        cap_list.append(cv2.VideoCapture(os.path.join('videos', 'cam_' + str(i).zfill(2) + '_20190203.mp4')))

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
    # print('the bbox coord is {0} and its length {1}'.format(bbox_cv,len(bbox_cv))
    bbox = list([[int(bbox_cv[0]), int(bbox_cv[1]), int(bbox_cv[0] + bbox_cv[2]), int(bbox_cv[1] + bbox_cv[3])], ])
    print('the bbox coord is {0} and its length {1}'.format(bbox_cv, len(bbox_cv)))
    previous_bbox = bbox
    colors = (randint(64, 255), randint(64, 255), randint(64, 255))

    feature_set = []
    # print('bbox given to lyj {0}'.format(bbox))
    # print('the type of bbox {0}'.format(type(bbox)))
    init_feature, match_bbox = REID.rank_feature(bbox, frame, feature_set, init_frame=True)
    feature_set.append(init_feature)

    counter = 0
    denominator = 2

    while cap.isOpened():

        success, frame = cap.read()
        if not success:
            break
        if counter %  denominator == 0:
            match_bbox_list = det.detect_one_frame(frame)

            index = bbox_trick(previous_bbox,match_bbox_list)
            if index < 0:
                break

            match_bbox = match_bbox_list[index]
            previous_bbox = match_bbox


            p1 = (int(match_bbox[0]), int(match_bbox[1]))
            p2 = (int(match_bbox[2]), int(match_bbox[3]))
            cv2.rectangle(frame, p1, p2, colors[0], 2, 1)

            # show frame
            cv2.imwrite(os.path.join('./results', str(counter).zfill(5) + '.jpg'), frame)

            if min(p1) < 10 or max(p1) > (w - 10) or min(p2) < 0 or max(p2) > (h - 10):
                # model = model.cuda()
                feature_new = []
                bbox_new = []

                confidence_candidate = []
                bbox_candidate = []
                feature_candidate = []

                confidence_max = 0.6
                # needs multi threads here
                for i in range(1, len(video_all_paths) - 1):
                    cap = cap_list[i]
                    print('the {0} of the videos'.format(i))
                    success, frame = cap.read()
                    if not success:
                        break

                    bbox_list = det.detect_one_frame(frame)
                    match_feature_list, match_bbox_list, confidence_list = REID.rank_feature(bbox_list, frame,
                                                                                             feature_set,
                                                                                             init_frame=False)

                    print('the confidence list is {0}'.format(confidence_list))
                    # index = confidence_list.index(max(confidence_list))
                    index = bbox_trick(previous_bbox,match_bbox_list)

                    if index < 0:
                        index = confidence_list.index(max(confidence_list))

                    match_feature = match_feature_list[index]
                    match_bbox = match_bbox_list[index]
                    previous_bbox = match_bbox
                    confidence = confidence_list[index]

                    # feature_candidate += match_feature
                    # bbox_candidate += match_bbox
                    # confidence_candidate += confidence_list

                    if confidence > confidence_max:
                        confidence_max = confidence
                        feature_new = match_feature
                        bbox_new = match_bbox
                        camID = i

                cap = cap_list[camID]
                feature_set.append(match_feature)

                print(bbox_new)
                p1 = (int(bbox_new[0]), int(bbox_new[1]))
                p2 = (int(bbox_new[2]), int(bbox_new[3]))
                cv2.rectangle(frame, p1, p2, colors[0], 2, 1)
                # show frame
                cv2.imwrite(os.path.join('./results', str(counter).zfill(5) + '.jpg'), frame)
        counter += 1


if __name__ == '__main__':
    main()