import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import time
import math

def update_w_and_b(X, y, w, b, alpha):
  '''Actualiza los parámetros w y b durante un epochs'''
  dl_dw = 0.0  # Inicializa la derivada parcial de la función de pérdida con respecto a w
  dl_db = 0.0  # Inicializa la derivada parcial de la función de pérdida con respecto a b
  N = len(X)   # Número de datos en el conjunto X
  for i in range(N):
    y_pred = w * X[i] + b  # Predicción de y usando los valores actuales de w y b
    error = y[i] - y_pred  # Calcula el error entre la predicción y el valor real
    dl_dw += -2*X[i]*(error)  # Calcula la derivada parcial con respecto a w
    dl_db += -2*(error)       # Calcula la derivada parcial con respecto a b
  # Actualiza los valores de w y b
  w = w - (1/float(N))*dl_dw*alpha  # Actualiza w usando la tasa de aprendizaje alpha
  b = b - (1/float(N))*dl_db*alpha  # Actualiza b usando la tasa de aprendizaje alpha

  return w, b  # Devuelve los valores actualizados de w y b

def train(X, y, w, b, alpha, epochs):
  '''Entrena el modelo durante múltiples epochs y muestra el progreso'''
  print('Progreso del entrenamiento:')
  for e in range(epochs):
    w, b = update_w_and_b(X, y, w, b, alpha)  # Actualiza w y b en cada epoch
    # Muestra el progreso cada 400 epochs
    if e % 400 == 0:
      avg_loss_ = avg_loss(X, y, w, b)  # Calcula la pérdida promedio (error cuadrático medio)
      print("epoch {} | Pérdida: {} | w:{}, b:{}".format(e, avg_loss_, round(w, 4), round(b, 4)))
  return w, b  # Devuelve los valores finales de w y b

def avg_loss(X, y, w, b):
  '''Calcula el error cuadrático medio (MSE)'''
  N = len(X)  # Número de datos en el conjunto X
  total_error = 0.0  # Inicializa el error total
  for i in range(N):
    total_error += (y[i] - (w*X[i] + b))**2  # Suma el error cuadrado de cada predicción
  return total_error / float(N)  # Devuelve el error cuadrático medio

def predict(x, w, b):
  '''Realiza una predicción para un valor x dado'''
  return w*x + b  # Devuelve la predicción usando los valores de w y b

# Carga los datos desde un archivo CSV
df = pd.read_csv('2019.csv')
print(df.head())  # Muestra las primeras filas del DataFrame

# Elimina filas con valores NaN en las columnas "GDP per capita" y "Score"
df = df.dropna(subset=["GDP per capita", "Score"]).reset_index(drop=True)

# Asigna las columnas a las variables x e y
x = df["GDP per capita"]
y = df["Score"]

# Grafica los datos
df.plot(x="GDP per capita", y="Score", kind="scatter")
plt.show()

# Inicializa los parámetros del modelo
w = 0.0
b = 0.0
alpha = 0.001  # Tasa de aprendizaje
epochs = 16000  # Número de epochs

# Entrena el modelo
w, b = train(X=x, y=y, w=w, b=b, alpha=alpha, epochs=epochs)

# Grafica la recta de regresión
plt.plot(x, w*x + b, color='red')
plt.xlabel('GDP per capita')
plt.scatter(x, y, color='blue', label='Datos')
plt.ylabel('Score')
plt.title('Regresión lineal simple')
plt.show()

# Realiza una predicción para un valor específico
print("Valor predicho para 1.5: ", predict(1.5, w, b))
