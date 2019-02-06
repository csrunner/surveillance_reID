# -*- coding:utf-8 -*-
__author__ = 'shichao'

import cv2
import os

import sys
from random import randint
import glob


import detection


def main():
    camID = 0

    det = detection.Detection(0, './cfg/yolov3.cfg', './cfg/yolov3.weights')

    # fetch the selected video
    video_init_path = "videos/cam_00_20190203.mp4"
    # Create a video capture object to read videos
    cap = cv2.VideoCapture(video_init_path)


    # success, frame = cap.read()
    # h,w,c = frame.shape
    # if not success:
    #     print('Failed to read video')
    #     sys.exit(1)

    colors=(randint(64, 255), randint(64, 255), randint(64, 255))
    counter = 0

    while cap.isOpened():

        success, frame = cap.read()
        if not success:
            break
        bbox_list = det.detect_one_frame(frame)

        for i, newbox in enumerate(bbox_list):
            p1 = (int(newbox[0]), int(newbox[1]))
            p2 = (int(newbox[2]), int(newbox[3]))
            cv2.rectangle(frame, p1, p2, colors[0], 2, 1)

        # show frame
        cv2.imwrite(os.path.join('./detection_results', str(counter).zfill(5) + '.jpg'), frame)


        counter += 1

if __name__ == '__main__':
    main()
