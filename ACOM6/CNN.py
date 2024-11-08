import keras
import numpy as np
from tensorflow.keras import layers
from keras import Sequential
# Загружаем и разделяем набор данных на обучающую и тестовую выборки.
mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Предобработка данных (преобразовываем все значения пикселя в диапазон от 0(черный), 1(белый))
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Изменение размерности данных
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# Сверточная нейронная сеть с одним сверточным слоем и слоем субдискритизации(уменьшение размерности)
def One_CNN():
    model = Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1))) # Сверточный слой, 32-нейрона, (3, 3) - ядро свертки, для получения карт признаков, input_shape=(28, 28, 1) изображение (28, 28, чб)
    model.add(layers.MaxPooling2D(pool_size=(2, 2))) # Уменьшаем размер карты признаков, после свертки, снижение выч затрат и переобучение
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    return model

model1 = One_CNN()
model1.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model1.fit(x_train, y_train, epochs=5, batch_size=32)

test_loss1, test_acc1 = model1.evaluate(x_test, y_test)
print(f'Точность (One_CNN): {test_acc1}')


# Сверточная нейронная сеть с двумя сверточными слоями и слоями субдискритизации
def Two_CNN():
    model = Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    return model

model2 = Two_CNN()
model2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model2.fit(x_train, y_train, epochs=5, batch_size=32)

test_loss2, test_acc2 = model2.evaluate(x_test, y_test)
print(f'Точность (Two_CNN): {test_acc2}')

def Three_CNN():
    model = Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    return model

model3 = Three_CNN()
model3.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model3.fit(x_train, y_train, epochs=5, batch_size=32, verbose=1)

test_loss3, test_acc3 = model3.evaluate(x_test, y_test)
print(f'Точность (Three_CNN): {test_acc3}')