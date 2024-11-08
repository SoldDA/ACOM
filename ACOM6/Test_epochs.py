import time
import keras
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# Загружаем и разделяем набор данных на обучающую и тестовую выборки.
mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Предобработка данных (преобразовываем все значения пикселя в диапазон от 0(черный), 1(белый))
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Изменение размерности данных
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# Создание модели многослойного персептрона
# Функции активации определяют как результат каждого нейрона будет использоваться в следующем слое
# Функция активации relu - если входное значение (x) положительное, вывод равен этому значению; если отрицательное — вывод ноль. Проста и быстра в вычисляемости
# Функция активации softmax - Эта функция преобразует необработанные выходные данные в вероятности, которые суммируются до 1. Тобишь вероятность каждой из 10 цифр будет в диапазоне от 0 до 1.
def train_model(epoch):
    model = keras.Sequential([
        layers.Flatten(input_shape=(28, 28)),  # Преобразование 2D изображения размером 28 на 28 в 1D(вектор, длина которого 784)
        layers.Dense(128, activation='relu'),   # Первый скрытый слой(полносвязный) Каждый нейрон связан со всеми нейронами предыдущего слоя и применяется функция активации к входным данным
        layers.Dense(64, activation='relu'),    # Второй скрытый слой
        layers.Dense(10, activation='softmax')   # Выходной слой для 10 классов(цифры 0 до 9)
    ])

    # Компиляция модели - задаем основные параметры и функции, которые будет использовать модель в обучении
    # optimizer='adam' - используется для того, чтобы находить минимальное значение функции потерь.
    # loss='sparse_categorical_crossentropy' - это метод оценки, который измеряет, насколько хорошо модель делает предсказания по сравнению с фактическими значениями.
    # metrics=['accuracy'] - метрика необходима для оценки производительности модели(точность)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    start_time = time.time()

    # Обучение модели
    # batch_size=32 - это количество изображений(примеров), которые используются для обновления весов модели за раз
    model.fit(x_train, y_train, epochs=epoch, batch_size=32)

    end_time = time.time()

    # Оценка модели на тестовых данных
    test_loss, test_acc = model.evaluate(x_test, y_test)
    training_time = end_time - start_time
    return test_acc, training_time

epochs = [5, 10, 15, 20]
results = []

for value in epochs:
    accuracy, training_time = train_model(value)
    results.append((value, accuracy, training_time))
    print(f"Эпох: {value}, Точность: {accuracy:.4f}, Время обучения: {training_time:.4f} секунд")

# # Сохранение модели
# model.save("mnist_model.h5")