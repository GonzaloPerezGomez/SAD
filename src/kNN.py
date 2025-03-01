# -*- coding: utf-8 -*-
"""
Autor: Xabier Gabiña Barañano
Script para la implementación del algoritmo kNN
Recoge los datos de un fichero csv y los clasifica en función de los k vecinos más cercanos
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def preprocesadoKNN(datos: pd.DataFrame, conf: pd.DataFrame):

    #Pasar las features categoriales a numericas
    encoder = LabelEncoder()
    cf = conf["preprocessing"]["categorical_features"]
    nf = conf["preprocessing"]["numerical_features"]
    
    for feature in cf:
        datos[feature] = encoder.fit_transform(datos[feature])

    f = cf + nf

    #Tratamiento de missing values
    mv = conf["preprocessing"]["missing_values"]
    ist = conf["preprocessing"]["impute_strategy"]

    if mv == "drop":
        datos = datos.dropna()
    elif mv == "impute":
        if ist == "mean":
            for feature in f:
                datos[feature] = datos[feature].fillna(datos[feature].mean())
        if ist == "median":
            for feature in f:
                datos[feature] = datos[feature].fillna(datos[feature].median())
        if ist == "mode":
            for feature in f:
                datos[feature] = datos[feature].fillna(datos[feature].mode())

    exit()
    return datos
  
def kNN(data, k, weights, p, conf):
    """
    Función para implementar el algoritmo kNN
    
    :param data: Datos a clasificar
    :type data: pandas.DataFrame
    :param k: Número de vecinos más cercanos
    :type k: int
    :param weights: Pesos utilizados en la predicción ('uniform' o 'distance')
    :type weights: str
    :param p: Parámetro para la distancia métrica (1 para Manhattan, 2 para Euclídea)
    :type p: int
    :return: Clasificación de los datos
    :rtype: tuple
    """

    # Seleccionamos las características y la clase
    X = data.iloc[:, :-1].values # Todas las columnas menos la última
    y = data.iloc[:, -1].values # Última columna
    
    # Dividimos los datos en entrenamiento y test
    from sklearn.model_selection import train_test_split
    np.random.seed(42)  # Set a random seed for reproducibility
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    
    # Escalamos los datos
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    # Entrenamos el modelo
    from sklearn.neighbors import KNeighborsClassifier
    classifier = KNeighborsClassifier(n_neighbors = k, weights = weights, p = p)
    classifier.fit(X_train, y_train)
    
    # Predecimos los resultados
    y_pred = classifier.predict(X_test)
    
    return y_test, y_pred, classifier