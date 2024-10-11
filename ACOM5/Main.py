import cv2

def main(path_to_file, kernel_size, standart_deviation, treshLow, min_area):

    # Задание 1 (самостоятельно). Реализовать метод, который читает
    # видеофайл и записывает в один файл только ту часть видео, где в кадре было
    # движение, можно воспользоваться примерами.

    video = cv2.VideoCapture(path_to_file, cv2.CAP_ANY)
    ret, frame = video.read()

    # Получаем ширину и высоту видео в пикселях из видеопотока
    # Используем метод get объекта видеозахвата video,
    # чтобы получить значение конкретного свойства видеопотока
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Указываем кодек видео
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')

    # Записываем видео в файл с частотой кадров - 25 и размером кадров
    # Частота кадров - это кол-во кадров, отображаемых в секунду при воспроизведении видео
    outputVideo = cv2.VideoWriter("new_test_video.mp4", fourcc, 25, (width, height))

    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    oldFrame = cv2.GaussianBlur(gray, (kernel_size, kernel_size), standart_deviation)
    while True:
        ret, frame = video.read()
        # если чтение неуспешно, остановить цикл
        if not ret:
            break

        # Теперь текущий кадр необходимо перевести в оттенки серого и применяем размытие по Гауссу
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurGray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), standart_deviation)

        # Вычисление разницы между кадрами
        frame_diff = cv2.absdiff(oldFrame, blurGray)

        # Применяем бинаризацию, чтобы выделить измененные области
        thresh = cv2.threshold(frame_diff, treshLow, 255, cv2.THRESH_BINARY)[1]

        # Находим контуры в бинарном изображении
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        try:
            contour = contours[0] # Получаем контур из контуров
            area = cv2.contourArea(contour) # Вычисляем площадь контура
            if area > min_area:
                outputVideo.write(frame) # Если площадь больше порога записываем текущий кадр в выходное видео
        except:
            pass

        oldFrame = blurGray
    outputVideo.release()

# Задание 2 (самостоятельно). Провести эксперименты, выбирая
# различные значения параметров: размытие Гаусса, граница разделения для
# метода threshold, площадь минимального объекта, подобрать оптимальные
# значения параметров для данного видео.

main("test_video.mp4", 3, 60, 60, 20)