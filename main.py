import cv2

from src.person_announ.video_edit import display_video, get_video_window
from src.person_announ.image_edit import ImageEditing
from src.person_announ.face_detection import face_detection
from src.person_announ.args import CAMERA_FORMAT, VIDEO_FORMAT, VIDEO_PATH

windowSize = [2000, 1000]
# TODO сделать более точную детекцию, тк в DNN она легла.
# TODO Зарефакторить ошибку при отсутствии лиц
# TODO Добавить детекцию лиц с помощью HOG
# TODO Добавить несколько вариантов блюра ROI


def person_announcement(model: str,
                        window_size: list[int]=(640, 480),
                        format: str=CAMERA_FORMAT,
                        video_path: str='VIDEO_PATH',
                       ):
    cap = get_video_window(*window_size)

    while True: 
        success, img = cap.read()
        if not success:
            raise Exception("Ошибка в захвате кадра") 

        img_edit = ImageEditing(image=img)
        img_edit.detect_preparation()
        # window_size = [img_edit.width, img_edit.height / 2]
        
        faces = face_detection(det_model=model, img=img_edit.img)
        if faces is None:
            raise Exception(f"Ошибка детекции лиц моделью (faces[1] = {faces})")

        for iteration, face in enumerate(faces):
            if model=='opencv_dnn':
                face_procent = face[14]
                main_points = face[4:-1]
                face = [int(item) for item in face[:4]]
            
            img_edit.blur_roi(dim=face, blur=31)
            img_edit.display_simple_interface(dim=face, dims=faces, iter=iteration)

        cv2.imshow('video', img_edit.img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    # person_announcement(model='haarcascade')
    person_announcement(model='opencv_dnn',
                        window_size=windowSize,
                        )
