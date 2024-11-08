import tkinter as tk
import cv2

from tkinter import Label, Button, Frame
from PIL import Image, ImageTk

from src.person_announ.video_edit import display_video, get_video_window
from src.person_announ.image_edit import ImageEditing
from src.person_announ.face_detection import face_detection


def process_frame(frame):
    faces = face_detection('haarcascade', frame)

    # Рисуем прямоугольники вокруг обнаруженных лиц
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    return frame

def update_frame():
    success, frame = cap.read()
    if success:
        # Обрабатываем кадр для распознавания лиц
        frame = ImageEditing(frame)
        frame.detect_preparation()
        faces = face_detection('haarcascade', frame.img)
        if faces is not None:
            for iteration, face in enumerate(faces):
                frame.display_simple_interface(dim=face,
                                               dims=faces,
                                               iter=iteration)
        frame = process_frame(frame.img)
        
        # Преобразуем кадр из BGR в RGB для Tkinter
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(Image.fromarray(frame))
        
        # Обновляем Label с новым кадром
        lbl_video.config(image=img)
        lbl_video.image = img
    
    # Планируем следующий вызов
    root.after(10, update_frame)

def start_stream():
    """Запускает видеопоток."""
    global running
    running = True
    update_frame()

def stop_stream(event=None):
    """Останавливает видеопоток."""
    global running
    running = False
    cap.release()
    root.destroy()

# Создаём главное окно Tkinter
root = tk.Tk()
root.title("Распознавание лиц")

# Разделяем окно на две горизонтальные части
frame_info = Frame(root, bg="lightgray", width=200)
frame_video = Frame(root, bg="black")

# Устанавливаем пропорции
frame_info.pack(side='left', fill='both', expand=True)
frame_video.pack(side='right', fill='both', expand=True)

# Добавляем элементы в верхнюю часть (информация и кнопка)
Label(frame_info, text="Информация о видеопотоке", bg="lightgray", font=("Helvetica", 16)).pack(pady=10)
Button(frame_info, text="Запустить видеопоток", command=start_stream).pack(pady=10)

# Добавляем виджет Label для отображения видеопотока
lbl_video = Label(frame_video, bg="black")
lbl_video.pack(fill="both", expand=True)

# Настроим разделение в пропорции 1:3
root.columnconfigure(0, weight=1)  # Левая часть - 1
root.columnconfigure(1, weight=3)  # Правая часть - 3

cap = cv2.VideoCapture(0)

# Привязываем клавишу 'q' к завершению программы
root.bind('q', stop_stream)

# Запускаем главный цикл
root.mainloop()

# Освобождаем ресурсы
cap.release()
cv2.destroyAllWindows()