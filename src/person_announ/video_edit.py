import cv2

from image_edit import ImageEditing
from enums import *


def get_video_window(width: int, height: int):
    cap = cv2.VideoCapture(0)
    # устанавливаем размеры выводимого видео
    # propid==3 - width, propid==4 - height
    cap.set(3, width)
    cap.set(4, height)
    return cap


def display_video(vid_type: str, video_path: str = ''):
    cap = None
    if vid_type == 'video':
        cap = cv2.VideoCapture(video_path)
    elif vid_type == 'camera':
        cap = get_video_window(640, 480)

    while True:
        # первая переменная указывает на успешность захвата кадра, а вторая - сам кадр
        success, img = cap.read()
        frame = ImageEditing(image=img)
        frame.detect_preparation()
    
        cv2.imshow('video', frame.img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


# пример работы
if __name__ == '__main__':
   
    cap = cv2.VideoCapture(0)
    # устанавливаем размеры выводимого видео
    # propid==3 - width, propid==4 - height
    cap.set(3, 640)
    cap.set(4, 480)

    while True:
        # первая переменная указывает на успешность захвата кадра, а вторая - сам кадр
        success, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        binary = cv2.Canny(blur, 5, 125, 250)
        
        cv2.imshow('video', binary)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break