import cv2
import numpy as np
import os

from enums import IMAGE_PATH

black = np.zeros((300, 300, 3), np.uint8)
white = np.ones((300, 300, 3), np.uint8) * 255
# С помощью среза списка, изменяем цвет каждого пикселя на (115, 54, 99)
black[100:250, 50:100] = 115, 54, 99

square = cv2.rectangle(black, (50, 50), (100, 100), (50, 50, 50), 5)


img = square
cv2.imshow('image', img)
cv2.waitKey(0)