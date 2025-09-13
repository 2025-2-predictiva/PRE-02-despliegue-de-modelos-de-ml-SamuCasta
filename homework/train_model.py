## train_model.py
# Este script construye, entrena y guarda un modelo de regresión lineal usando scikit-learn

"""
Construye, despliega y accede a un modelo usando scikit-learn
"""

import pickle  # Para guardar el modelo entrenado en un archivo

import pandas as pd  # Para manipulación de datos
from sklearn.linear_model import LinearRegression  # Para crear el modelo de regresión lineal

df = pd.read_csv("files/input/house_data.csv", sep=",")  # Carga los datos de las casas desde un archivo CSV

features = df[
    [
        "bedrooms",      # Número de habitaciones
        "bathrooms",     # Número de baños
        "sqft_living",   # Metros cuadrados habitables
        "sqft_lot",      # Metros cuadrados del lote
        "floors",        # Número de pisos
        "waterfront",    # Si tiene vista al agua
        "condition",     # Condición de la casa
    ]
]  # Selecciona las columnas que serán usadas como variables predictoras

target = df[["price"]]  # Selecciona la columna objetivo (precio de la casa)

estimator = LinearRegression()  # Crea el modelo de regresión lineal
estimator.fit(features, target)  # Entrena el modelo con los datos

with open("homework/house_predictor.pkl", "wb") as file:
    pickle.dump(estimator, file)  # Guarda el modelo entrenado en un archivo para su uso posterior