import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return x * (1 - x)

# Генерація значень x від -7 до 7 з кроком 0.1
x_values = np.arange(-7, 7, 0.1)

# Обчислення відповідних значень сигмоїди для кожного x
y_values = sigmoid(x_values)

# Побудова графіка
plt.plot(x_values, y_values, label='Сигмоїда')
plt.title('Графік функції сигмоїди')
plt.xlabel('x')
plt.ylabel('sigmoid(x)')
plt.legend()
plt.grid(True)
plt.show()
