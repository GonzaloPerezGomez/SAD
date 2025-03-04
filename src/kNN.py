# -*- coding: utf-8 -*-
"""
Autor: Xabier Gabiña Barañano
Script para la implementación del algoritmo kNN
Recoge los datos de un fichero csv y los clasifica en función de los k vecinos más cercanos
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


#---------------------------------------------------------------------------------------------------------------------------------#

def preprocesadoKNN(datos: pd.DataFrame, conf: pd.DataFrame):
    #TODO: Añadir procesado de undersampling/oversampling

    #Pasar las features de text a vectores numericos
    tf = conf["preprocessing"]["text_features"]
    tp = conf["preprocessing"]["text_process"]

    #si json tenia almacenada una columna de texto
    if len(tf) != 0:
        #por cada categoria que esta como texto
        for feature in tf:
            #si se ha escogido BOW o one_hot
            if tp == "BOW" or tp == "one_hot":
                #crea un vector binario/frecuencia por cada frase
                v = CountVectorizer()
            #si ha escogido tf-idf
            elif tp == "tf-idf":
                #crea vectores como el BOW pero estos tienen peso en base a su frecuencia en las frases
                v = TfidfVectorizer()
            
            # Aplicar la transformación de los datos 
            transformed = v.fit_transform(datos[feature].fillna(""))

            if tp == "one_hot":
                transformed[transformed>1] = 1

            # Convertir a DataFrame y agregar nombres de columnas
            df_transformed = pd.DataFrame(transformed.toarray(), 
                                            columns=[f"{feature}_{i}" for i in range(transformed.shape[1])])
                
            # Eliminar la columna original y añadir las nuevas
            datos = datos.drop(columns=[feature]).join(df_transformed)

    #Pasar las features categoriales a numericas
    encoder = LabelEncoder()
    cf = conf["preprocessing"]["categorical_features"]
    nf = conf["preprocessing"]["numerical_features"]
    
    for feature in cf:
        datos[feature] = encoder.fit_transform(datos[feature])

    if len(tf) != 0:
        f = cf + nf + df_transformed.columns.tolist()
    else:
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
            min_value = datos[feature].min()
            max_value = datos[feature].max()
            
            if min_value != max_value:
                datos[feature] = (datos[feature] - min_value) / (max_value - min_value)
            else:
                print(f"Atributo {feature} no se ha escalado porque min y max son iguales.")

    elif e == "avgstd":
        for feature in f:
            avg = datos[feature].mean()
            std = datos[feature].std()
            
            if std != 0:
                datos[feature] = (datos[feature] - avg) / std
            else:
                print(f"Atributo {feature} no se ha escalado porque su desviación estándar es 0.")

    #Rebalanceo

    return datos


#---------------------------------------------------------------------------------------------------------------------------------#

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