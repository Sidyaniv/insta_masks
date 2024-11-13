import cv2

from src.person_announ.video_edit import display_video, get_video_window
from src.person_announ.image_edit import ImageEditing
from src.person_announ.face_detection import face_detection
from src.person_announ.utils import preparate_points, get_avg_face_point
from src.person_announ.args import CAMERA_FORMAT, VIDEO_FORMAT, VIDEO_PATH

windowSize = [700, 700]


# TODO Аннотация типов
# TODO Оформить документацию
# TODO маски 2д

def person_announcement(model: str,
                        window_size: list[int] = (640, 480),
                        blur_type: str = 'median',
                        magnitude: int = 15,
                        # format: str = CAMERA_FORMAT,
                        # video_path: str = 'VIDEO_PATH',
                        ):
    cap = get_video_window(*window_size)

    while True:
        success, img = cap.read()
        if not success:
            raise Exception("Ошибка при захвате кадра")

        img_edit = ImageEditing(image=img)
        img_edit.detect_preparation()

        faces = face_detection(det_model=model, img=img_edit.img)

        if faces is not None:
            for iteration, face in enumerate(faces):
                if model == 'yunet':
                    face = [round(item) for item in face[:14]]
                    face_bbox, key_points = face[:4], face[4:]
                    img_edit.draw_key_points(key_points)

                    avg_poi = get_avg_face_point(key_points)

                    # Код для слежки за лицом
                    # x, y , width, height = img_edit.correct_dimension(face_bbox)

                    # x_left = round(avg_poi[0] - (width * 0.9))
                    # x_right = round(x_left + width * 1.8)
                    # y_left = round(avg_poi[1] - (height * 0.9))
                    # y_right = round(y_left + height * 1.8)
                    # img_edit.img = img_edit.img[y_left:y_right, x_left:x_right]
                else:
                    face_bbox = face

                img_edit.blur_roi(type=blur_type,
                                  dim=face_bbox,
                                  magnitude=magnitude)
                img_edit.display_simple_interface(dim=face_bbox,
                                                  dims=faces,
                                                  iter=iteration)

        cv2.imshow('video', img_edit.img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    # person_announcement(model='haarcascade',
    # window_size=windowSize,
    # blur_type='pixelization',
    # )

    person_announcement(model='yunet',
                        window_size=windowSize,
                        blur_type='pixelization',
                        magnitude=12
                        )

