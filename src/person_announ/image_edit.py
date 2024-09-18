import cv2
import numpy as np

from src.person_announ.enums import IMAGE_PATH


def show_img(img) -> None:
    cv2.imshow('image', img)
    cv2.waitKey(0)


class ImageEditing():
    def __init__(self, image_path: str = None, image: np.ndarray = None):
        if image_path is not None:
            self.image_path = image_path
            self.img = cv2.imread(image_path)
        else:
            self.img = image

        self.height, self.width = self.img.shape[:2]

    # new_img = np.zeros((, 3), np.uint8)

    def resize_img(self, width=200, height=2000):
        self.img = cv2.resize(self.img, (width, height))
        # выводим часть изображения с помощью срезов
        # return new_img[0:200, 0:200, ::-1]

    def gauss_blur(self):
        img = cv2.GaussianBlur(self.img, (9, 9), 10)

        return img

    def bgr2gray(self):
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        return img

    def rotate_image(self, angle):
        height, width = self.img.shape[:2]
        rot_poi = (width / 2, height / 2)
        matrix = cv2.getRotationMatrix2D(rot_poi, angle, 1)
        self.img = cv2.warpAffine(self.img, matrix, (width, height))

    def transform_image(self, x, y):
        matrix = np.float32([[1, 0, x], [0, 1, y]])
        self.img = cv2.warpAffine(self.img, matrix, (self.width, self.height))

    def get_contours(self):
        con, hir = cv2.findContours(self.img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        return con

    def gray2binary(self):
        self.img = cv2.Canny(self.img, 100, 100)


# пример работы
if __name__ == '__main__':
    people = ImageEditing(image_path=IMAGE_PATH)
    people.bgr2gray()
    people.img = cv2.GaussianBlur(people.img, (5, 5), 0)
    people.gray2binary()
    contours = people.get_contours()

    show_img(people.img)
