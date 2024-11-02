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


    def correct_dimension(self, dim: list): 
        if dim is not None:
            # x, y = map(dim[:2], lambda x: x if x >= 0 else 0)
            x, y , width, height = dim
            if x < 0:
                width += -x 
                x = 0
            if y < 0:
                height += -y 
                y = 0
            return x, y , width, height
        else:
            raise Exception("Dimension error")


    def detect_preparation(self):
        '''Подготовка изображения к детектированию лиц'''
        cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        cv2.GaussianBlur(self.img, (9, 9), 10)


    def blur_roi(self, type: str, dim: list, magnitude: int=15):
        x, y , width, height = self.correct_dimension(dim)

        roi = self.img[y:y+height, x:x+width]

        if type == 'blur':
            blur_roi = cv2.blur(self.img[y:y+height, x:x+width])
        elif type == 'median':
            blur_roi = cv2.medianBlur(self.img[y:y+height, x:x+width], magnitude)
        elif type == 'pixelization':
            temp = cv2.resize(roi, (width // magnitude, 
                                    height // magnitude), 
                                    interpolation=cv2.INTER_AREA)
            blur_roi = cv2.resize(temp, 
                                 (width, height), 
                                 interpolation=cv2.INTER_NEAREST)
        # else:
            # raise Exception("Anonymization error")
        self.img[y:y+height, x:x+width] = blur_roi


    def display_simple_interface(self,
                             dim: list,
                             dims: list,
                             iter: int = 0,
                            ):
        '''Отображает bbox лица и его номер среди всех найденных лиц
        Param:
            dim (list) - Координаты левого верхнего угла bbox 
                и длина его рёбер[x, y, width, height]
            iter (int) - номер лица
        '''
        x, y, width, height = self.correct_dimension(dim)
        window_size = (self.width // 16, self.height // 10)
        cv2.rectangle(self.img, (x, y), (x + width, y + height), (255, 0, 0), 2)
        cv2.putText(self.img, 
                    f"Face_{iter + 1}", 
                    (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    (0, 255, 0),
                    2)
        cv2.putText(self.img,
                        f"Detected faces: {len(dims)}",
                        window_size,
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1, 
                        (0, 0, 255),
                        3)


    def resize_img(self, width=200, height=2000):
        self.img = cv2.resize(self.img, (width, height))
        # выводим часть изображения с помощью срезов
        # return new_img[0:200, 0:200, ::-1]


    def rotate_image(self, angle: int):
        '''angle (int) - angle of rotate in degrees'''
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
    pass
    # people = ImageEditing(image_path=IMAGE_PATH)
    # show_img(people.img)



    # def display_hard_interface(self,
                        #   dim: list,
                        #   iter: int = 0
                        #  ):
        #    '''Отображает bbox лица, координаты левого верхнего угла bbox и длина его рёбер, номер лица, вероятность детекции лица.
        #    Param:
            #    dim (list) - Координаты левого верхнего угла bbox 
                #    и длина его рёбер[x, y, width, height]
            #    iter (int) - номер лица
        #    Return: 
            #    -
        #    '''
        #    x, y, width, height = dim
# 
        #    cv2.rectangle(self.img, (x, y), (x + width, y + height), (255, 0, 0), 2)
        #    cv2.putText(self.img, f"Face_{iter}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))


    # def median_blur_roi(self, dim: list, blur: int):
    #     '''Анонимизация region of interest путём блюра изображения
    #     Param:
    #         dim (list) - Координаты левого верхнего угла roi
    #             и длина его рёбер[x, y, width, height]
    #     '''
    #     x, y, width, height = self.correct_dimension(dim)
    #     # Чтобы избежать ошибки с отрицательными срезами:
    #     # idx_x = x if x >= 0 else idx_x = 0
    #     # idx_y = y if y >= 0 else idx_y = 0
    #     roi = self.img[y:y+height, x:x+width]
    #     blur_roi = cv2.medianBlur(roi, blur)
    #     self.img[y:y+height, x:x+width] = blur_roi


    # def pixelization_roi(self, dim: list, pix: int):
    #     '''Анонимизация region of interest путём пикселизации изображения
    #     Param:
    #         dim (list) - Координаты левого верхнего угла roi
    #             и длина его рёбер[x, y, width, height]
    #     '''
    #     x, y , width, height = self.correct_dimension(dim)

    #     roi = self.img[y:y+height, x:x+width]
        
    #     temp = cv2.resize(roi, (width // pix, height // pix), interpolation=cv2.INTER_AREA)
    #     pix_roi = cv2.resize(temp, (width, height), interpolation=cv2.INTER_NEAREST)

    #     self.img[y:y+height, x:x+width] = pix_roi
