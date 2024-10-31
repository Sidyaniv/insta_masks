import cv2
import os

from src.person_announ.video_edit import display_video, get_video_window
from src.person_announ.image_edit import ImageEditing
from src.person_announ.face_detection import face_detection
from src.person_announ.args import CAMERA_FORMAT, VIDEO_FORMAT, VIDEO_PATH


def person_announcement(model: str,
                        format: str=CAMERA_FORMAT,
                        video_path: str='VIDEO_PATH',
                       ):
    cap = get_video_window(640, 480)

    while True: 
        # первая переменная указывает на успех захвата кадра, а вторая - сам кадр
        success, img = cap.read()
        if not success:
            raise Exception("Ошибка в захвате кадра") 

        img_edit = ImageEditing(image=img)
        img_edit.detect_preparation()
        
        faces = face_detection(det_model=model, img=img_edit.img)

        if faces is not None:
            print(f"Обнаруженные лица: {faces}", "", sep='\n')
        else: 
            raise Exception(f"Ошибка детекции лиц моделью (faces[1] = {faces})")

        for iteration, face in enumerate(faces):
            if model=='opencv_dnn':
                face_procent = face[14]
                print(f"Вероятность правильного обнаружения лица: {face_procent * 100}%")
                face = [int(item) for item in face[:4]]

            img_edit.blur_roi(dim=face, blur=31)
            img_edit.display_simple_interface(dim=face,iter=iteration)

        cv2.imshow('video', img_edit.img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    # person_announcement(model='haarcascade')
    person_announcement(model='opencv_dnn')
