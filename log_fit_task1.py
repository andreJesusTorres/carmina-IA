import pandas as pd  # Librería para manejo de datos tipo tabla (DataFrames)
import numpy as np  # Librería para operaciones numéricas y matemáticas
import matplotlib.pyplot as plt  # Librería para graficar
from sklearn.linear_model import LinearRegression  # Para ajustar modelos lineales
from sklearn.metrics import r2_score  # Para calcular el coeficiente de determinación R^2

# 1. Cargar los datos desde un archivo CSV
filename = 'p3_task-1.csv'  # Nombre del archivo de datos
# Lee el archivo CSV y lo guarda en un DataFrame de pandas
data = pd.read_csv(filename)

# Asumimos que las columnas se llaman 'x' y 'y'. Si no, ajusta aquí:
x = data['x'].values.reshape(-1, 1)  # Extrae la columna 'x' y la convierte en vector columna
# (reshape(-1, 1) es necesario para scikit-learn)
y = data['y'].values  # Extrae la columna 'y' como vector

# 2. Graficar los datos originales
plt.scatter(x, y, label='Datos originales')  # Dibuja los puntos (x, y)
plt.xlabel('x')  # Etiqueta del eje x
plt.ylabel('y')  # Etiqueta del eje y
plt.title('Datos originales')  # Título del gráfico
plt.legend()  # Muestra la leyenda
plt.show()  # Muestra la gráfica

# 3. Transformar x usando logaritmo
x_log = np.log(x)  # Aplica logaritmo natural a cada valor de x

# 4. Ajustar el modelo lineal: y = a*log(x) + b
model = LinearRegression()  # Crea el modelo de regresión lineal
model.fit(x_log, y)  # Ajusta el modelo usando log(x) como variable independiente y y como dependiente

a = model.coef_[0]  # Obtiene el coeficiente 'a' (pendiente)
b = model.intercept_  # Obtiene el intercepto 'b' (ordenada al origen)
print(f"Modelo ajustado: y = {a:.4f} * log(x) + {b:.4f}")  # Imprime la ecuación ajustada

# 5. Graficar la curva ajustada junto a los datos
x_range = np.linspace(x.min(), x.max(), 200).reshape(-1, 1)  # Genera 200 valores de x entre el mínimo y máximo
# Calcula las predicciones del modelo para esos valores de x (usando log(x))
y_pred_curve = model.predict(np.log(x_range))

plt.scatter(x, y, label='Datos originales')  # Vuelve a graficar los puntos originales
plt.plot(x_range, y_pred_curve, color='red', label='Ajuste logarítmico')  # Dibuja la curva ajustada
plt.xlabel('x')  # Etiqueta del eje x
plt.ylabel('y')  # Etiqueta del eje y
plt.title('Ajuste logarítmico a los datos')  # Título del gráfico
plt.legend()  # Muestra la leyenda
plt.show()  # Muestra la gráfica

# 6. Calcular RSS y R^2
# Predicciones para los datos originales
y_pred = model.predict(x_log)  # Predice los valores de y usando el modelo ajustado
RSS = np.sum((y - y_pred) ** 2)  # Calcula la suma de los residuos al cuadrado (RSS)
R2 = r2_score(y, y_pred)  # Calcula el coeficiente de determinación R^2

print(f"RSS (Residual Sum of Squares): {RSS:.4f}")  # Imprime el RSS
print(f"R^2 (Coeficiente de determinación): {R2:.4f}")  # Imprime el R^2

# 7. Discusión del resultado
if R2 > 0.9:
    print("El ajuste es muy bueno (R^2 > 0.9)")  # Si R^2 es mayor a 0.9, el ajuste es excelente
elif R2 > 0.7:
    print("El ajuste es razonable (R^2 > 0.7)")  # Si R^2 es mayor a 0.7, el ajuste es aceptable
else:
    print("El ajuste no es muy bueno (R^2 <= 0.7)")  # Si R^2 es menor o igual a 0.7, el ajuste no es bueno 