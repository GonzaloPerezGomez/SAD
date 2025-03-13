import numpy as np
from sklearn.naive_bayes import GaussianNB


def trainNB(k, K, p, datos, file, conf):

    gen_informe_NB(True, date, top_k, top_p,top_w, top_fscore_micro, top_fscore_macro, top_prec, top_rec, file)
    return 

def gen_informe_NB(es_final, date, indexK, indexP, w, fscore_micro, fscore_macro, prec, rec, file: str):
    with open(f'informes/informeNB-{file}-{date}.csv', 'a', newline='') as csvfile:
        spamwriter = csv.writer(csvfile)
        if es_final:
            spamwriter.writerow(["Combinacion optima:"])
            
        

#--------------------------------------------------------------------------------------------------------------------------------#

def NB(data: pd.DataFrame, k, weights, p):
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
    classifier = GaussianNB()
    
    classifier.fit(X_train, y_train)
    
    # Predecimos los resultados
    y_pred = classifier.predict(X_test)
    
    return y_test, y_pred, classifier