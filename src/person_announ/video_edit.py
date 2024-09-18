import cv2

from image_edit import ImageEditing
from src.person_announ.enums import *


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
        cadr = ImageEditing(image=img)
        cadr.bgr2gray()
        cadr.gray2binary()
        cv2.imshow('video', cadr.img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


# пример работы
if __name__ == '__main__':
    get_video(VIDEO_FORMAT, VIDEO_PATH)