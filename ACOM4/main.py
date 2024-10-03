import cv2
import numpy as np
print(cv2.__version__)

# Нахождение округления угла между вектором градиента и осью Х
# Схема округления угла до 45 градусов
# Здесь принято в качестве угла использовать одно из значений 0 – 7,
# характеризующееся указанными на схеме границами значений частных
# производных по x и y и тангенсом угла градиента.
def get_angle_number(Gx, Gy, tg):
    if (Gx < 0):
        if (Gy < 0):
            if (tg > 2.414): return 0
            elif (tg < 0.414): return 6
            elif (tg <= 2.414): return 7
        else:
            if (tg < -2.414): return 4
            elif (tg < -0.414): return 5
            elif (tg >= -0.414): return 6
    else:
        if (Gy < 0):
            if (tg < -2.414): return 0
            elif (tg < -0.414): return 1
            elif (tg >= -0.414): return 2
        else:
            if (tg < 0.414): return 2
            elif (tg < 2.414): return 3
            elif (tg >= 2.414): return 4

def main(path, standart_deviation, kernel_size, porog):
    # Задание 1
    # Реализовать метод, который принимает в качестве строки
    # полный адрес файла изображения, читает изображение, переводит его в черно
    # белый цвет и выводит его на экран применяет размытие по Гауссу и выводит
    # полученное изображение на экран.
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    cv2.imshow("GrayImage", image)
    gaussianBlur = cv2.GaussianBlur(image, (kernel_size, kernel_size), standart_deviation)
    cv2.imshow("GaussianBlurImage", gaussianBlur)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Задание 2
    # Модифицировать построенный метод так, чтобы в результате
    # вычислялось и выводилось на экран две матрицы – матрица значений длин и
    # матрица значений углов градиентов всех пикселей изображения.

    # Градиент – это вектор, состоящий из двух значений. В градиентных методах используется простейший случай – длина
    # вектора.

    # если длина градиента яркости пикселя больше длины градиента соседей
    # и больше некоторой порогово величины, то данный пиксель считается границей

    length = np.zeros(gaussianBlur.shape) #  длины градиента
    angle = np.zeros(gaussianBlur.shape) # угла градиента

    for x in range(1, (len(gaussianBlur) - 1)):
        for y in range(1, len(gaussianBlur[0]) - 1):
            # Канни рассматривается оператор Собеля для вычисления частных производных
            Gx = (gaussianBlur[x + 1][y + 1] - gaussianBlur[x - 1][y - 1] +
                  gaussianBlur[x + 1][y - 1] - gaussianBlur[x - 1][y + 1] +
                  2 * (gaussianBlur[x + 1][y] - gaussianBlur[x - 1][y]))
            Gy = (gaussianBlur[x + 1][y + 1] - gaussianBlur[x - 1][y - 1] +
                  gaussianBlur[x - 1][y + 1] - gaussianBlur[x + 1][y - 1] +
                  2 * (gaussianBlur[x][y + 1] - gaussianBlur[x][y - 1]))
            length[x][y] = np.sqrt(Gx**2 + Gy**2) # находим длину вектора градиента
            tg = np.arctan(Gy / Gx) # Для этого найдем величину угла градиента
            print(Gx, Gy, length[x][y])
            get_angle_number(Gx, Gy, tg)

    cv2.imshow("Length", length)
    cv2.imshow("Angle", angle)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Задание 3. Модифицировать метод так, чтобы он выполнял подавление
    # немаксимумов и выводил полученное изображение на экран. Рассмотреть
    # изображение, сделать выводы.

    # ГРАНИЦЕЙ БУДЕТ СЧИТАТЬСЯ ПИКСЕЛЬ, ГРАДИЕНТ КОТОРОГО
    # МАКСИМАЛЕН В СРАВНЕНИИ С ПИКСЕЛЯМИ ПО НАПРАВЛЕНИЮ
    # НАИБОЛЬШЕГО РОСТА ФУНКЦИИ

    # ЕСЛИ ЗНАЧЕНИЕ ГРАДИЕНТА
    # ВЫШЕ, ЧЕМ У ПИКСЕЛЕЙ СЛЕВА И СПРАВА, ТО ДАННЫЙ ПИКСЕЛЬ –
    # ЭТО ГРАНИЦА, ИНАЧЕ – НЕ ГРАНИЦА.
    maxLen = np.max(length)
    borders = np.zeros(gaussianBlur.shape)
    for x in range(1, len(gaussianBlur) - 1):
        for y in range(1, len(gaussianBlur[0]) - 1):
            ix = 0
            iy = 0
            if (angle[x][y] == 0): iy = -1
            if (angle[x][y] == 1):
                iy = -1
                ix = 1
            if (angle[x][y] == 2): ix = 1
            if (angle[x][y] == 3):
                iy = 1
                ix = 1
            if (angle[x][y] == 4): iy = 1
            if (angle[x][y] == 5):
                iy = 1
                ix = -1
            if (angle[x][y] == 6): ix = -1
            if (angle[x][y] == 7):
                iy = -1
                ix = -1

            border = length[x][y] > length[x + ix][y + iy] and length[x][y] > length[x - ix][y - iy]
            borders[x][y] = 255 if border else 0
    cv2.imshow("Borders", borders)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    # Задание 4. Модифицировать метод так, чтобы он выполнял двойную
    # пороговую фильтрацию и выводил полученное изображение на экран.

    # Если значение градиента меньше нижней границы, то пиксель не граница, если значение
    # градиента выше-верхней границы, то пиксель точно граница.
    # После такого фильтра останутся пиксели, значение градиента которых
    # заключено между границами.

    # Если пиксель – это граница, то он не может быть отдельной
    # границей, рядом должен быть еще пиксель с границей.
    # Добавим проверку на то, что рядом с границей есть другая граница,
    # для чего необходимо проверить 8 пикселей вокруг заданного.

    low_level = maxLen // porog
    high_level = maxLen // porog

    for x in range(1, len(gaussianBlur) - 1):
        for y in range(1, len(gaussianBlur[0]) - 1):
            if ((borders[x][y] == 255) and (length[x][y] < low_level)): borders[x][y] = 0

    for x in range(1, len(gaussianBlur) - 1):
        for y in range(1, len(gaussianBlur[0]) - 1):
            if ((borders[x][y] == 255) and (length[x][y] <= high_level)):
                if (borders[x - 1][y - 1] == 255 or borders[x - 1][y] == 255 or borders[x - 1][y + 1] == 255 or borders[x][y + 1] == 255 or borders[x + 1][y + 1] == 255 or borders[x + 1][y] == 255 or borders[x + 1][y - 1] == 255 or borders[x][y - 1] == 255): borders[x][y] = 255
                else: borders[x][y] = 0

    cv2.imshow("TwoBordersFilter", borders)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

main("mem.jpg", 100, 5, 1)
#main("mem.jpg", 20, 7, 1)
#main("mem.jpg", 50, 5, 15)
#main("mem.jpg", 1000, 3, 9)
#main("mem.jpg", 100, 5, 1)