import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

mnist = keras.datasets.mnist
(_, _), (x_test, y_test) = mnist.load_data()

# Предобработка данных
x_test = x_test.astype("float32") / 255.0
x_test = np.expand_dims(x_test, axis=-1)

# Загрузка сохраненной модели
model = keras.models.load_model('mnist_model.h5')

# Получаем предсказания
predictions = model.predict(x_test)

# Функция для отображения предсказания
def display_prediction(image, true_label, predicted_label):
    plt.imshow(image.squeeze(), cmap='gray')
    plt.title(f'Цифра: {true_label}, Модель распознала: {predicted_label}')
    plt.axis('off')
    plt.show()

# Тестируем на первых 10 изображениях
for i in range(3):
    true_label = y_test[i]
    predicted_label = np.argmax(predictions[i])  # Индекс класса с максимальной вероятностью
    display_prediction(x_test[i], true_label, predicted_label)