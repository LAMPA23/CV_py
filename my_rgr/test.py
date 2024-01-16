import numpy as np

# Функція активації (сигмоїда)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Похідна від сигмоїди
def sigmoid_derivative(x):
    return x * (1 - x)

# Функція навчання
def train_neural_network(inputs, outputs, learning_rate, epochs):
    input_layer_size = len(inputs[0])
    output_size = len(outputs[0])

    # Ініціалізація ваг випадковими значеннями
    weights = 2 * np.random.random((input_layer_size, output_size)) - 1

    print(inputs)

    for epoch in range(epochs):
        # Прямий прохід (визначення виходу мережі)
        input_layer = inputs
        output_layer = sigmoid(np.dot(input_layer, weights))

        # Обчислення помилки
        error = outputs - output_layer

        print(outputs)
        print(output_layer)
        print(error)
        print('---------')

        # Зворотній прохід (корекція ваг)
        adjustment = error * sigmoid_derivative(output_layer)
        weights += np.dot(input_layer.T, adjustment) * learning_rate

    return weights

# Приклад вхідних та вихідних даних
inputs = np.array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
outputs = np.array([[0, 1, 1, 0]]).T

# Навчання нейромережі
learning_rate = 0.01
epochs = 10
trained_weights = train_neural_network(inputs, outputs, learning_rate, epochs)

# Тестування навченої мережі
new_inputs = np.array([[1, 0, 0]])
predicted_output = sigmoid(np.dot(new_inputs, trained_weights))

# Класифікація: якщо значення > 0.5, то клас 1, інакше клас 0
predicted_class = 1 if predicted_output > 0.5 else 0

print("Вхідні дані:", new_inputs)
print("Прогнозований вихід:", predicted_output)
print("Прогнозований клас:", predicted_class)
