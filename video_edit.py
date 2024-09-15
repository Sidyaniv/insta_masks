import cv2
import numpy as np
import os
from enums import *

videos_path = './videos/people_go.mp4'


def get_video(vid_type: str, video_path: str = ''):
    cap = None
    if vid_type == 'video':
        cap = cv2.VideoCapture(video_path)
    elif vid_type == 'camera':
        cap = cv2.VideoCapture(0)
        # устанавливаем размеры выводимого видео
        # propid==3 - width, propid==4 - height
        cap.set(3, 640)
        cap.set(4, 480)

    while True:
        # первая переменная указывает на успешность захвата кадра, а вторая - сам кадр
        success, img = cap.read()
        cv2.imshow('video', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


get_video(CAMERA_FORMAT, videos_path)