import cv2
import os

from src.person_announ.video_edit import display_video, get_video_window
from src.person_announ.image_edit import ImageEditing
from src.person_announ.args import CAMERA_FORMAT, VIDEO_FORMAT, VIDEO_PATH


haarcascade_path = os.path.join(os.getcwd(), 'models', 'haarcascade_frontalface_default.xml')
opencv_dnn_path = os.path.join(os.getcwd(), 'models', 'face_detection_yunet_2023mar_int8.onnx')
# prototxt_config = ""

def person_announcement(model: str,
                        format: str=CAMERA_FORMAT,
                        video_path: str='VIDEO_PATH',
                       ):
    cap = get_video_window(640, 480)
    # cap = cv2.VideoCapture("videotestsrc ! videoconvert ! appsink", cv2.CAP_V4L2)
    

    while True: 
        # первая переменная указывает на успех захвата кадра, а вторая - сам кадр
        success, img = cap.read()

        if not success:
            raise Exception('Ошибка в захвате кадра') 

        img_edit = ImageEditing(image=img)
        img_edit.detect_preparation()
        
        frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if model=='haarcascade':
            cascade = cv2.CascadeClassifier(haarcascade_path) 
            faces = cascade.detectMultiScale(img_edit.img, 
                                        scaleFactor=1.9, 
                                        minNeighbors=2)
        elif model=='opencv_dnn':
            # TODO: OpenCV DNN face detection
            detector = cv2.FaceDetectorYN.create(
                model=opencv_dnn_path,
                config="",
                input_size=((img_edit.width, img_edit.height)),
                score_threshold=0.9,
                nms_threshold=0.3,
                top_k=500,
                # backend_id=0,
                # target_id=0
            )
            detector.setInputSize((frameWidth, frameHeight))
            faces = detector.detect(img_edit.img)
            if faces[1] is not None:
                print(f"Обнаруженные лица: {faces}", "", sep='\n')
            else: 
                print("Не обнаружено лиц на потоковом видео")
            # tm = cv2.TickMeter()

        
        for iteration, dimension in enumerate(faces):
            # x, y, width, height = dimension
            
            img_edit.blur_roi(dim=dimension, blur=31)
            img_edit.display_simple_interface(dim=dimension,iter=iteration)

        cv2.imshow('video', img_edit.img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    # person_announcement(model='haarcascade')
    person_announcement(model='opencv_dnn')
