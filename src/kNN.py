# -*- coding: utf-8 -*-
"""
Autor: Xabier Gabiña Barañano
Script para la implementación del algoritmo kNN
Recoge los datos de un fichero csv y los clasifica en función de los k vecinos más cercanos
"""
import csv
from datetime import datetime
import numpy as np
import pandas as pd

import metrics 
import preprocesado


#---------------------------------------------------------------------------------------------------------------------------------#
#entrenamiento knn del modelo
def trainKNN(k, K, p, datos, file, conf):
    
    top_fscore_micro = 0
    weights = ["uniform", "distance"]

    #obtenemos la hora
    date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    #si no hay .json (preprocesado basico)
    if not(type(conf) is type(None)):
        datos = preprocesado.preprocesadoKNN(datos, conf)


    for indexK in range(k, K+1):
        for indexP in range(1, p+1):
            for w in weights:
                y_test, y_pred, model = kNN(datos, indexK, w, indexP)
                fscore_micro, fscore_macro = metrics.calculate_fscore(y_test, y_pred)
                prec, rec = metrics.calculate_prec_rec(y_test, y_pred)
                gen_informe_KNN(False, date, indexK, indexP, w, fscore_micro, fscore_macro, prec, rec, file)

                if fscore_micro > top_fscore_micro:
                    top_fscore_micro = fscore_micro
                    top_model = model
                    top_k = indexK
                    top_p = indexP
                    top_w = w
                    top_fscore_macro = fscore_macro
                    top_prec = prec
                    top_rec = rec

    gen_informe_KNN(True, date, top_k, top_p,top_w, top_fscore_micro, top_fscore_macro, top_prec, top_rec, file)
    return top_model


def gen_informe_KNN(es_final, date, indexK, indexP, w, fscore_micro, fscore_macro, prec, rec, file: str):
    with open(f'informes/informeKNN-{file}-{date}.csv', 'a', newline='') as csvfile:
        spamwriter = csv.writer(csvfile)
        if es_final:
            spamwriter.writerow(["Combinacion optima:"])
            
        spamwriter.writerow(["K = " + str(indexK), "P = " + str(indexP), w, "Micro = " + str(fscore_micro), 
                                "Macro = " + str(fscore_macro), "Precision = " + str(prec), "Recall = " + str(rec)])


def kNN(data: pd.DataFrame, k, weights, p):
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