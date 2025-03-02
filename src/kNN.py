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
    #TODO: Añadir preprocesado de textos (BOW, tf-idf)
    #TODO: Añadir procesado de undersampling/oversampling

    #Pasar las features categoriales a numericas
    encoder = LabelEncoder()
    cf = conf["preprocessing"]["categorical_features"]
    nf = conf["preprocessing"]["numerical_features"]
    
    for feature in cf:
        datos[feature] = encoder.fit_transform(datos[feature])

    f = cf + nf

    #Outliers
    o = conf["preprocessing"]["outliers"]
    ost = conf["preprocessing"]["outliers_strategy"]

    for feature in f:
        if ost == "quantile":
            Q1 = datos[feature].quantile(0.25)
            Q3 = datos[feature].quantile(0.75)
            IQR = Q3 - Q1

            limite_inferior = Q1 - 1.5 * IQR
            limite_superior = Q3 + 1.5 * IQR
        
        elif ost == "std":
            mean = datos[feature].mean()
            std = datos[feature].std()

            limite_inferior = mean - 3 * std
            limite_superior = mean + 3 * std

            datos[feature] = datos[feature].astype(float) #No es necesario pero numpy lo recomienda para avitar futuros problemas

        if o == "drop":
            datos = datos[((datos[feature]>=limite_inferior) & (datos[feature]<=limite_superior)) 
                          | (datos[feature].isna())]

        elif o == "round":
            datos.loc[datos[feature]<=limite_inferior, feature] = limite_inferior
            datos.loc[datos[feature]>=limite_superior, feature] = limite_superior

    #Tratamiento de missing values
    mv = conf["preprocessing"]["missing_values"]
    ist = conf["preprocessing"]["impute_strategy"]

    if mv == "drop":
        datos = datos.dropna()
        print(datos[datos.isna().any(axis=1)])
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


    ##Reescalado
    
    e = conf["preprocessing"]["scaling"]

    if e == "min-max":
        for feature in f:
            min = datos[feature].min()
            max = datos[feature].max()
            try:
                datos[feature] = (datos[feature] - min) / (max - min)
            except ZeroDivisionError:
                print(f"Atributo {feature} no se ha escalado dado que el min y el max es el mismo")

    elif e == "avgstd":
        for feature in f:
            avg = datos[feature].mean()
            std = datos[feature].std()
            try:
                datos[feature] = (datos[feature] - avg) / std
            except ZeroDivisionError:
                print(f"Atributo {feature} no se ha escalado dado que de su desviación es 0")

    print(datos[datos.isna().any(axis=1)])
    return datos
  
def kNN(data: pd.DataFrame, k, weights, p, conf):
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

    print(data[data.isna().any(axis=1)])
    # Seleccionamos las características y la clase
    X = data.iloc[:, :-1].values # Todas las columnas menos la última
    y = data.iloc[:, -1].values # Última columna
    
    # Dividimos los datos en entrenamiento y test
    from sklearn.model_selection import train_test_split
    np.random.seed(42)  # Set a random seed for reproducibility
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    
    # Entrenamos el modelo
    from sklearn.neighbors import KNeighborsClassifier
    classifier = KNeighborsClassifier(n_neighbors = k, weights = weights, p = p)
    
    classifier.fit(X_train, y_train)
    
    # Predecimos los resultados
    y_pred = classifier.predict(X_test)
    
    return y_test, y_pred, classifier