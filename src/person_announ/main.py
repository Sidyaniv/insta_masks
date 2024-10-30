import cv2

from video_edit import display_video, get_video_window
from image_edit import ImageEditing
from enums import CAMERA_FORMAT, VIDEO_FORMAT, VIDEO_PATH


def person_announcement(model: str,
                        format: str='camera',
                        video_path: str='../../videos/people_go.mp4'
                       ):
    cap = get_video_window(640, 480)

    while True: 
        # первая переменная указывает на успешность захвата кадра, а вторая - сам кадр
        success, img = cap.read()

        img_edit = ImageEditing(image=img)
        img_edit.detect_preparation()
        if model=='haarcascade':
            cascade = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml") 
            faces = cascade.detectMultiScale(img_edit.img, 
                                        scaleFactor=1.3, 
                                        minNeighbors=5)
        elif model=='DNN':
            pass
        
        for iter, dimension in enumerate(faces):
            x, y, width, height = dimension

            img_edit.blur_roi(dim=dimension, blur=31)
            img_edit.draw_faces_interface(dim=dimension)

        cv2.imshow('video', img_edit.img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    person_announcement(model='haarcascade')
