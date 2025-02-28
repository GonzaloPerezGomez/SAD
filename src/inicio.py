
import csv
import pandas as pd
import kNN
from datetime import datetime

def pedir_param():
    file = input("De que archivo quieres carga el dataset: ")
    k = int(input("Numero minimo de vecinos(k): "))
    K = int(input("Numero maximo de vecinos(k): "))
    p = int(input("Numero maximo de p (1,2): "))

    if k <= 0 or K <= 0:
        print("Los numeros de vecinos deben ser positivos")
        exit()
    elif K < k:
        print("El numero maximo de vecinos no puede ser mayor al numero minimo de vecinos")
        exit()
    elif p != 1 and p != 2:
        print("El valor de p debe ser 1 o 2")
        exit()

    return k, K, p, file

def load_data(file):
    """
    Función para cargar los datos de un fichero csv
    :param file: Fichero csv
    :return: Datos del fichero
    """
    data = pd.read_csv(file)
    return data

def calculate_fscore(y_test, y_pred):
    """
    Función para calcular el F-score
    :param y_test: Valores reales
    :param y_pred: Valores predichos
    :return: F-score (micro), F-score (macro)
    """
    from sklearn.metrics import f1_score
    fscore_micro = f1_score(y_test, y_pred, average='micro')
    fscore_macro = f1_score(y_test, y_pred, average='macro')
    return fscore_micro, fscore_macro

def calculate_confusion_matrix(y_test, y_pred):
    """
    Función para calcular la matriz de confusión
    :param y_test: Valores reales
    :param y_pred: Valores predichos
    :return: Matriz de confusión
    """
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    return cm

def calculate_prec_rec(y_test, y_pred):

    from sklearn.metrics import precision_score, recall_score
    return precision_score(y_test, y_pred, average="weighted"), recall_score(y_test, y_pred, average="weighted")

def train(k, K, p, datos):
    
    top_fscore = 0
    weights = ["uniform", "distance"]
    date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    for indexK in range(k, K+1):
        for indexP in range(1, p+1):
            for w in weights:
                y_test, y_pred, model = kNN.kNN(datos, indexK, w, indexP)
                fscore_micro, fscore_macro = calculate_fscore(y_test, y_pred)
                prec, rec = calculate_prec_rec(y_test, y_pred)
                gen_informe(date, indexK, indexP, w, fscore_micro, fscore_macro, prec, rec)

                if fscore_micro > top_fscore:
                    top_fscore = fscore_micro
                    top_model = model
    return top_model

def gen_informe(date, indexK, indexP, w, fscore_micro, fscore_macro, prec, rec):
    with open(f'informes/informeKNN{date}.csv', 'a', newline='') as csvfile:
        spamwriter = csv.writer(csvfile)
        spamwriter.writerow(["K = " + str(indexK), "P = " + str(indexP), w, "Micro = " + str(fscore_micro), 
                             "Macro = " + str(fscore_macro), "Precision = " + str(prec), "Recall = " + str(rec)])

def guardar_modelo(modelo):

    import pickle 
    
    nombreModel = f"models/TopModelKNN{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.sav" 
    saved_model = pickle.dump(modelo, open(nombreModel,'wb'))
    print('Modelo guardado correctamente empleando Pickle')

if __name__ == "__main__":

    k, K, p, file = pedir_param()

    try:
        datos = load_data(file)
    except FileNotFoundError:
        print("Fichero no encontrado, asegurate de escribir bien la ruta")
        exit()

    top_model = train(k, K, p, datos)

    guardar_modelo(top_model)


