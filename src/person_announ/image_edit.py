import cv2
import numpy as np


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


    def detect_preparation(self):
        '''Подготовка изображения к детектированию лиц
        Param:
            -
        Return: 
            -
        '''
        cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        cv2.GaussianBlur(self.img, (9, 9), 10)


    def blur_roi(self, dim: list, blur: int):
        '''Анонимизация region of interest путём блюра изображения
        Param:
            dim (list) - Координаты левого верхнего угла roi
                и длина его рёбер[x, y, width, height]
        Return: 
            -
        '''
        x, y , width, height = dim

        roi = self.img[y:y+height, x:x+width]
        blur_roi = cv2.medianBlur(self.img[y:y+height, x:x+width], blur)
        self.img[y:y+height, x:x+width] = blur_roi


    def display_simple_interface(self,
                             dim: list,
                             iter: int = 0
                            ):
        '''Отображает bbox лица
        Param:
            dim (list) - Координаты левого верхнего угла bbox 
                и длина его рёбер[x, y, width, height]
            iter (int) - номер лица
        Return: 
            -
        '''
        x, y, width, height = dim

        cv2.rectangle(self.img, (x, y), (x + width, y + height), (255, 0, 0), 2)
        cv2.putText(self.img, f"Face_{iter}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))


    def display_interface(self,
                          dim: list,
                          iter: int = 0
                         ):
           '''Отображает bbox лица, координаты левого верхнего угла bbox и длина его рёбер, номер лица, вероятность детекции лица.
           Param:
               dim (list) - Координаты левого верхнего угла bbox 
                   и длина его рёбер[x, y, width, height]
               iter (int) - номер лица
           Return: 
               -
           '''
           x, y, width, height = dim

           cv2.rectangle(self.img, (x, y), (x + width, y + height), (255, 0, 0), 2)
           cv2.putText(self.img, f"Face_{iter}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))


    def resize_img(self, width=200, height=2000):
        self.img = cv2.resize(self.img, (width, height))
        # выводим часть изображения с помощью срезов
        # return new_img[0:200, 0:200, ::-1]


    def rotate_image(self, angle: int):
        '''
        angle (int) - angle of rotate in degrees
        '''
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



# пример работы
if __name__ == '__main__':
    people = ImageEditing(image_path=IMAGE_PATH)
    # people.bgr2gray()
    people.img = cv2.GaussianBlur(people.img, (51,51), 0)
    # people.gray2binary()
    # contours = people.get_contours()

    show_img(people.img)
