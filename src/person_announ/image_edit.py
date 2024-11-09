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
        if magnitude  % 2 == 0:
            magnitude += 1

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

            # resize делается для предотвращения исключения вызванного 
            # изменениями размерности roi при его пикселизации 
            blur_roi = cv2.resize(blur_roi, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST)
        else:
            raise Exception("Anonymization type error")
        
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

        text = f"Detected faces: {len(dims)}"
        detect_fsize = cv2.getTextSize(text,
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1,
                                    3)[0]
        cv2.putText(self.img,
                    text,
                    (window_size[0], 2 * detect_fsize[1] ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 3)
        
        text = f"Face_{iter + 1}"
        face_fsize = cv2.getTextSize(text, 
                                     cv2.FONT_HERSHEY_SIMPLEX, 
                                     1, 
                                     2)[0]
        # Резиновая верстка 
        top_location = y - 10
        bottom_location = y + dim[3] + 2 * face_fsize[1]
        cv2.putText(self.img, 
                    text, 
                    (x, top_location if y > 3 * detect_fsize[1] + face_fsize[1] else bottom_location), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 255, 0), 2)
        
        
        cv2.rectangle(self.img, (x, y), (x + width, y + height), (255, 0, 0), 2)


    def draw_key_points(self, coordinates: list):
        for i in range(0, len(coordinates), 2):
            cv2.circle(self.img, 
                       (coordinates[i], coordinates[i+1]), 
                       2, (0, 0, 255), 2)
                    #    2, (0, 0, i * 30), 2)



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

        #    cv2.rectangle(self.img, (x, y), (x + width, y + height), (255, 0, 0), 2)
        #    cv2.putText(self.img, f"Face_{iter}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
