import cv2

from src.person_announ.video_edit import display_video, get_video_window
from src.person_announ.image_edit import ImageEditing
from src.person_announ.face_detection import face_detection
from src.person_announ.args import CAMERA_FORMAT, VIDEO_FORMAT, VIDEO_PATH

windowSize = [700, 700]
# TODO Добавить детекцию лиц с помощью HOG
# TODO Создать общую функцию блюра и установить тип блюра
    #  параметром в главную функцию
# TODO Убрать баг в пикселизации с изменением размерности roi
# TODO Протестить баг с отрицательными значениями roi
# TODO Оформить документацию

def person_announcement(model: str,
                        window_size: list[int]=(640, 480),
                        format: str=CAMERA_FORMAT,
                        video_path: str='VIDEO_PATH',
                       ):
    cap = get_video_window(*window_size)

    while True: 
        success, img = cap.read()
        if not success:
            raise Exception("Ошибка при захвате кадра") 

        img_edit = ImageEditing(image=img)
        img_edit.detect_preparation()
        # window_size = [img_edit.width, img_edit.height / 2]
        
        faces = face_detection(det_model=model, img=img_edit.img)
        # if faces is None:
            # raise Exception(f"Ошибка детекции лиц моделью (faces[1] = {faces})")
        if faces is not None:
            for iteration, face in enumerate(faces):
                if model=='opencv_dnn':
                    face_procent = face[14]
                    face = [round(item) for item in face[:14]]
                    main_points = face[4:]

                    for iter in range(0, len(main_points), 2):
                        cv2.circle(img_edit.img, 
                                (main_points[iter], main_points[iter + 1]), 
                                1, (0, 255, 0), 2)
                    face = face[:4]                
                        
                img_edit.blur_roi(type='pixelization', dim=face)
                # img_edit.median_blur_roi(dim=face, blur=31)
                # img_edit.pixelization_roi(dim=face, pix=15)
                

                img_edit.display_simple_interface(dim=face, dims=faces, iter=iteration)

        cv2.imshow('video', img_edit.img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    # person_announcement(model='haarcascade',
                        # window_size=windowSize,
                        # )

    person_announcement(model='opencv_dnn',
                        window_size=windowSize,
                        )
