import cv2
import numpy as np

from src.person_announ.image_edit import show_img

# !Внимание! Когда мы рисуем на изображении, старые фигуры не удаляются!
# мы изменяем изображение img если не передаём в функцию img.copy(

black = np.zeros((300, 900, 3), np.uint8)
white = np.ones((300, 300, 3), np.uint8) * 255
# С помощью среза списка, изменяем цвет каждого пикселя на (115, 54, 99)
black[100:250, 50:100] = 115, 54, 99
# квадрат
square_1 = cv2.rectangle(black, (5, 5), (10, 100), (50, 50, 50), 2)
print(black)
square_2 = cv2.rectangle(black.copy(), (50, 50), (100, 100), (50, 50, 50), thickness=cv2.FILLED)

# линия
line_1 = cv2.line(white.copy(), (50, 50), (100, 100), (100, 100, 100), 5)

# круг
circle_1 = cv2.circle(white.copy(), (250, 150), 50, (200, 200, 200), 1)
circle_2 = cv2.circle(white.copy(), (150, 150), 50, (200, 200, 200), cv2.FILLED)

# ввод текста
# 1. фото, 2. текст, 3. координаты, 4. шрифт, 5. размер шрифта, 6. цвет, 7. толщина
text = cv2.putText(white, 'falling', (200, 200), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)

# вывод рисунка

show_img(white)
