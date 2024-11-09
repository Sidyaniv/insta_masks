import cv2
import numpy as np
import os


haarcascade_path = os.path.join(os.getcwd(), 'models', 'haarcascade_frontalface_default.xml')
opencv_dnn_path = os.path.join(os.getcwd(), 'models', 'face_detection_yunet_2023mar_int8.onnx')


def face_detection(det_model: str, img: np.ndarray):
    if det_model=='haarcascade':
        cascade = cv2.CascadeClassifier(haarcascade_path) 
        faces = cascade.detectMultiScale(img, 
                                         scaleFactor=1.3, 
                                         minNeighbors=5)
    elif det_model=='opencv_dnn':
        height, width = img.shape[:2]
        
        detector = cv2.FaceDetectorYN.create(
            model=opencv_dnn_path,
            config="",
            input_size=((width, height)),
            score_threshold=0.85,
            nms_threshold=0.3,
            top_k=500,
        )
        detector.setInputSize((width, height))
        _, faces = detector.detect(img)
    return faces