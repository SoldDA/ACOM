import cv2
import numpy as np

# cap = cv2.VideoCapture(0)
# while True:
#     ret, frame = cap.read()
#     # преобразование изображения в формат HSV
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#     cv2.imshow("HSV_image", hsv)
#     cv2.imwrite("DANIL.png", hsv)
#     if cv2.waitKey(1) & 0xFF == 27:
#         break
#
# cap.release()
# cv2.destroyAllWindows()


# cap = cv2.VideoCapture(0)
# while True:
#     ret, frame = cap.read()
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#
#     # Определение диапазона красного цвета в HSV
#     lower_red = np.array([0, 48, 80])  # Минимальные значения оттенка, насыщенности и яркости
#     upper_red = np.array([20, 255, 255])  # Максимальные значения оттенка, насыщенности и яркости
#
#     # Маска - бинарное изображение, в которой соответствуют заданному диапазону цвета, а именно значение 255 (белый), а остальные имеют значение 0 (черный)
#     mask = cv2.inRange(hsv, lower_red, upper_red)
#
#     # Операцию "И" между пикселями исходного изображения (frame) и маской (mask)
#     res = cv2.bitwise_and(frame, frame, mask=mask)
#
#     cv2.imshow('HSV', hsv)
#     cv2.imshow('Result', res)
#
#     if cv2.waitKey(1) & 0xFF == 27:
#         break
#
# cap.release()
# cv2.destroyAllWindows()


# # erode - уменьшение изображения для удаления мелких деталей и шумов
# def erode(image, kernel):
#     m, n, _ = image.shape  # Получаем высоту и ширину изображения
#     km, kn = kernel.shape
#     hkm = km // 2
#     hkn = kn // 2
#     eroded = np.copy(image)
#
#     # Проходимся по каждому пикселю изображения, начиная с пикселей, где размер ядра помещается полностью
#     for i in range(hkm, m - hkm):
#         for j in range(hkn, n - hkn):
#             # Вычисляем минимум среди пикселей внутри ядра, только если соответствующий элемент ядра равен 1
#             eroded[i, j] = np.min(
#               #создания подматрицы (подобласти) изображения вокруг текущего пикселя
#                 image[i - hkm :i + hkm + 1, j - hkn :j + hkn + 1][kernel == 1])
#     return eroded
#
# # dilate - увеличение изображения для заполнения недостающих частей
# def dilate(image, kernel):
#     m, n, _ = image.shape  # Получаем высоту и ширину изображения
#     km, kn = kernel.shape
#     hkm = km // 2
#     hkn = kn // 2
#     dilated = np.copy(image)
#
#     # Проходимся по каждому пикселю изображения, начиная с пикселей, где размер ядра помещается полностью
#     for i in range(hkm, m - hkm):
#         for j in range(hkn, n - hkn):
#             # Вычисляем максимум среди пикселей внутри ядра, только если соответствующий элемент ядра равен 1
#             dilated[i, j] = np.max(
#                 image[i - hkm:i + hkm + 1, j - hkn:j + hkn + 1][kernel == 1])
#     return dilated
#
# cap = cv2.VideoCapture(0)
# while True:
#     ret, frame = cap.read()
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#
#     # определение диапазона красного цвета в HSV
#     lower_red = np.array([0, 0, 100])  # минимальные значения оттенка, насыщенности и яркости
#     upper_red = np.array([60, 255, 255])  # максимальные значения оттенка, насыщенности и яркости
#     mask = cv2.inRange(hsv, lower_red, upper_red)
#
#     res = cv2.bitwise_and(frame, frame, mask=mask)
#
#     # структурирующий элемент (ядро) размером 5x5, который представляет собой матрицу, где все элементы установлены в 1
#     # (определяет размер и форму области)
#     kernel = np.ones((5, 5), np.uint8)
#
#     # открытие - 1)эрозия, 2)дилатация.
#     opening = cv2.morphologyEx(res, cv2.MORPH_OPEN, kernel)
#
#     # закрытие - 1)дилатации, 2) операция эрозии.
#     closing = cv2.morphologyEx(res, cv2.MORPH_CLOSE, kernel)
#
#     cv2.imshow('Opening', opening)
#     cv2.imshow('Closing', closing)
#     if cv2.waitKey(1) & 0xFF == 27:
#         break
#
# cap.release()
# cv2.destroyAllWindows()


cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # определение диапазона красного цвета в HSV
    lower_red = np.array([0, 100, 100])  # минимальные значения оттенка, насыщенности и значения (яркости)
    upper_red = np.array([8, 255, 255])  # максимальные значения оттенка, насыщенности и значения (яркости)
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # Выполнить морфологические операции для разделения близких объектов
    kernel = np.ones((5, 5), np.uint8)  # Создать ядро (прямоугольник) размером 5x5
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # Применить операцию открытия (убрать мелкие шумы)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Применить операцию закрытия (заполнить дыры)

    # поиск контуров в бинарном изображении(точки, обозначающие границы объекта на изображении)
    # cv2.RETR_EXTERNAL - выделение только наружных границы объектов
    # cv2.CHAIN_APPROX_SIMPLE - контуры аппроксимируются с минимальным количеством точек для сохранности памяти
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Отрисовка контуров в кадре и вычисление моментов
    for contour in contours:
        area = cv2.contourArea(contour)  # Вычисление площади контура
        if area > 100:  # Фильтрация маленьких контуров по площади
            moments = cv2.moments(contour)  # Вычисление моментов контура

            # Нахождение координат центра объекта
            c_x = int(moments["m10"] / moments["m00"])
            c_y = int(moments["m01"] / moments["m00"])
            width = height = int(np.sqrt(area))
            cv2.rectangle(frame, (c_x - (width), c_y - (height)), (c_x + (width), c_y + (height)), (0, 0, 0), 2)

    cv2.imshow('HSV_frame', hsv)
    cv2.imshow('Result_frame', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()