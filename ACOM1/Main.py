import os
import cv2
import numpy as np

# image_path = ["one.jpg", "two.png", "three.bmp"]
# window_flags = [cv2.WINDOW_NORMAL, cv2.WINDOW_FULLSCREEN, cv2.WINDOW_AUTOSIZE]
# read_flags = [cv2.IMREAD_COLOR, cv2.IMREAD_GRAYSCALE, cv2.IMREAD_UNCHANGED]
# names_flag = ["Color", "Grayscale", "Unchanged"]
#
# for i, path in enumerate(image_path):
#     window_name = f"Image - {os.path.basename(path)}"
#     cv2.namedWindow(window_name, window_flags[i % len(window_flags)])
#
#     img = cv2.imread(path, read_flags[i % len(read_flags)])
#     cv2.imshow(window_name, img)
#     key = cv2.waitKey(0)
#
#     if key == ord('q'):
#         break
#
# cv2.destroyAllWindows()


# video_path = "MK11.mp4"
# cap = cv2.VideoCapture(video_path)
#
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fps = cap.get(cv2.CAP_PROP_FPS)
#
# print(f"Размер видео: {width}x{height}")
# print(f"FPS: {fps}")
#
# cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Original", width, height)
#
# while True:
#     ret, frame = cap.read()
#     cv2.imshow("Original", frame)
#
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     cv2.imshow("Grayscale", gray)
#
#     small_frame = cv2.resize(frame, (int(width * 0.5), int(height * 0.5)))
#     cv2.imshow("Small", small_frame)
#
#     big_frame = cv2.resize(frame, (int(width * 2), int(height * 2)), interpolation=cv2.INTER_CUBIC)
#     cv2.imshow("big", big_frame)
#
#     blur = cv2.GaussianBlur(frame, (5, 5), 0)
#     cv2.imshow("Blur", blur)
#
#     key = cv2.waitKey(1) & 0xFF
#     if key == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()


# input_video_path = "MK11.mp4"
# output_video_path = "output_MK11.mp4"
#
# cap = cv2.VideoCapture(input_video_path)
#
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fps = cap.get(cv2.CAP_PROP_FPS)
#
# fourcc = cv2.VideoWriter.fourcc(*'mp4v')
# out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
# success, frame = cap.read() ## Покадровая запись
#
# while success:
#     out.write(frame)
#     success, frame = cap.read()
#
# cap.release()
# out.release()


# image_path = "one.jpg"
# img = cv2.imread(image_path)
#
# cv2.namedWindow("HSV", cv2.WINDOW_NORMAL)
# cv2.namedWindow("ORIGINAL", cv2.WINDOW_NORMAL)
#
# hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#
# cv2.imshow("HSV", hsv_img)
# cv2.imshow("ORIGINAL", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# cap = cv2.VideoCapture(0)
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#
# while True:
#     ret, frame = cap.read()
#     matrix = np.zeros((height, width, 3), dtype=np.uint8)
#
#     center_x = width // 2
#     center_y = height // 2
#     ## Гор. палки
#     cv2.line(matrix, (center_x - 100, center_y - 20), (center_x + 100, center_y - 20), (0, 0, 255), 2)
#     cv2.line(matrix, (center_x - 100, center_y + 20), (center_x + 100, center_y + 20), (0, 0, 255), 2)
#     ## Вер. палки
#     cv2.line(matrix, (center_x - 20, center_y - 100), (center_x - 20, center_y + 100), (0, 0, 255), 2)
#     cv2.line(matrix, (center_x + 20, center_y - 100), (center_x + 20, center_y + 100), (0, 0, 255), 2)
#     ## Гор. мел. палки
#     cv2.line(matrix, (center_x - 20, center_y - 100), (center_x + 20, center_y - 100), (0, 0, 255), 2)
#     cv2.line(matrix, (center_x - 20, center_y + 100), (center_x + 20, center_y + 100), (0, 0, 255), 2)
#     ## Вер. мел. палки
#     cv2.line(matrix, (center_x - 100, center_y - 20), (center_x - 100, center_y + 20), (0, 0, 255), 2)
#     cv2.line(matrix, (center_x + 100, center_y - 20), (center_x + 100, center_y + 20), (0, 0, 255), 2)
#
#     result = cv2.addWeighted(frame, 1, matrix, 0.5, 0)
#     cv2.imshow("Camera", result)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()


# cap = cv2.VideoCapture(0)
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fps = cap.get(cv2.CAP_PROP_FPS)
#
# print(f"Разрешение видео: {width}x{height}")
# print(f"Частота кадров: {fps}")
#
# fourcc = cv2.VideoWriter.fourcc(*'mp4v')
# out = cv2.VideoWriter("CAM_Output.mp4", fourcc, fps, (width, height))
#
# while True:
#     ret, frame = cap.read()
#
#     cv2.imshow("CAM", frame)
#     out.write(frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# out.release()
# cv2.destroyAllWindows()


# # Загрузка изображения
# img = cv2.imread('one.jpg')
#
# # Создание окна отображения
# cv2.namedWindow('Display window', cv2.WINDOW_NORMAL)
#
# # Параметры для прямоугольников и линии
# color = (0, 0, 255)  # Красный цвет в BGR
# thickness = 2  # Толщина линии
#
# # Получение размеров изображения (ширина, высота)
# height, width, _ = img.shape
#
# # Координаты прямоугольников
# rect_width_1 = 50
# rect_height_1 = 400
# x1_1 = width // 2 - rect_width_1 // 2
# y1_1 = height // 2 - rect_height_1 // 2
# x2_1 = width // 2 + rect_width_1 // 2
# y2_1 = height // 2 + rect_height_1 // 2
#
# rect_width_2 = 50
# rect_height_2 = 350
# x1_2 = width // 2 - rect_height_2 // 2
# y1_2 = height // 2 - rect_width_2 // 2
# x2_2 = width // 2 + rect_height_2 // 2
# y2_2 = height // 2 + rect_width_2 // 2
#
# # Размер ядра для размытия
# kernel_size = (71, 11)
#
# # Выделение части изображения, соответствующей горизонтальному прямоугольнику
# img_part = img[y1_2:y2_2, x1_2:x2_2]
#
# # Применение размытия к выделенной части изображения
# img_part_blur = cv2.GaussianBlur(img_part, kernel_size, 30)
#
# # Замена исходной части изображения размытой версией
# img[y1_2:y2_2, x1_2:x2_2] = img_part_blur
#
# # Определение цвета центрального пикселя
# center_pixel = img[height//2][width//2]
# r, g, b = center_pixel
#
# # Список возможных цветов в формате RGB
# colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
#
# # Расчет расстояний до каждого цвета и выбор ближайшего цвета
# distances = []
# for color in colors:
#     distance = np.linalg.norm(np.array(color) - np.array([r, g, b]))
#     distances.append(distance)
#
# min_index = min(range(len(distances)), key=distances.__getitem__)
# nearest_color = colors[min_index]
#
# # Закрашивание прямоугольников ближайшим цветом
# cv2.rectangle(img, (x1_1, y1_1), (x2_1, y2_1), nearest_color, -1)
# cv2.rectangle(img, (x1_2, y1_2), (x2_2, y2_2), nearest_color, -1)
#
# # Отображение изображения
# cv2.imshow('Display window', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# Создание объекта VideoCapture для подключения к IP-камере
#URL-адрес потока видео с IP-камеры
cap = cv2.VideoCapture("http://10.191.196.111:8080/video")

while True:
    # Считывание кадра с IP-камеры
    ret, frame = cap.read()

    if ret:
        # Отображение кадра с IP-камеры на экране
        cv2.imshow("Phone's camera", frame)

        # Ожидание нажатия клавиши 'esc' для выхода из цикла
        if cv2.waitKey(1) & 0xFF == 'q':
            break
    else:
        # Если возникла ошибка при чтении видео, выходим из цикла
        print("Ошибка чтения видео")
        break

# Освобождение ресурсов и закрытие окон
cap.release()
cv2.destroyAllWindows()