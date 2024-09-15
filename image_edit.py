import cv2
import numpy as np
import os

from enums import IMAGE_PATH


def show_img(img) -> None:
    cv2.imshow('image', img)
    cv2.waitKey(0)


class ImageEditing:
    def __init__(self, image_path: str):
        self.image_path = image_path
        self.img = cv2.imread(image_path)

    def resize_img(self, width=200, height=2000):
        new_img = cv2.resize(self.img, (width, height))
        # выводим часть изображения с помощью срезов
        return new_img[0:200, 0:200, ::-1]

    def gauss_blur(self):
        img = cv2.GaussianBlur(self.img, (9, 9), 10)

        return img

    def rgb2gray(self):
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        return img

    def gray2binary(self):
        img = cv2.Canny(self.img, 200, 200)

        return img


people = ImageEditing(IMAGE_PATH)
# people.get_image()
# people.resize_img()
# people.gauss_blur()
# people.rgb2gray()
result_img = people.gray2binary()
show_img(result_img)