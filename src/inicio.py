
import csv
import pandas as pd
import kNN
from datetime import datetime

def pedir_param():

    algoritmo = int(input("Selecciona el algoritmo a usar(KNN = 0, DT = 1)"))

    if algoritmo == 0:
        
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

        conf = input("Indique el fichero json para el preprocesado(dejar en blanco si no requiere preprocesado)")

        return [algoritmo, file, k, K, p, conf]

    elif algoritmo == 1:
        #TODO: Definir alforitmo de arbol
        exit()
    
    else:
        print("Valor no valido")
        exit()

def load_data(file):
    """
    Función para cargar los datos de un fichero csv
    :param file: Fichero csv
    :return: Datos del fichero
    """
    data = pd.read_csv(file)
    return data

def load_json(file):
    """
    Función para cargar los datos de un fichero csv
    :param file: Fichero csv
    :return: Datos del fichero
    """
    if file != "":
        data = pd.read_json(file)
    else:
        data = None
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

def trainKNN(k, K, p, datos, file, conf):
    
    top_fscore_micro = 0
    weights = ["uniform", "distance"]
    date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    if not(type(conf) is type(None)):
        datos = kNN.preprocesadoKNN(datos, conf)

    for indexK in range(k, K+1):
        for indexP in range(1, p+1):
            for w in weights:
                y_test, y_pred, model = kNN.kNN(datos, indexK, w, indexP, conf)
                fscore_micro, fscore_macro = calculate_fscore(y_test, y_pred)
                prec, rec = calculate_prec_rec(y_test, y_pred)
                gen_informe(False, date, indexK, indexP, w, fscore_micro, fscore_macro, prec, rec, file)

                if fscore_micro > top_fscore_micro:
                    top_fscore_micro = fscore_micro
                    top_model = model
                    top_k = indexK
                    top_p = indexP
                    top_w = w
                    top_fscore_macro = fscore_macro
                    top_prec = prec
                    top_rec = rec

    gen_informe(True, date, top_k, top_p,top_w, top_fscore_micro, top_fscore_macro, top_prec, top_rec, file)
    return top_model

def gen_informe(es_final, date, indexK, indexP, w, fscore_micro, fscore_macro, prec, rec, file: str):
    with open(f'informes/informeKNN-{file}-{date}.csv', 'a', newline='') as csvfile:
        spamwriter = csv.writer(csvfile)
        if not es_final:
            spamwriter.writerow(["K = " + str(indexK), "P = " + str(indexP), w, "Micro = " + str(fscore_micro), 
                                "Macro = " + str(fscore_macro), "Precision = " + str(prec), "Recall = " + str(rec)])
        else:
            spamwriter.writerow(["Conbinacion optima:"])
            spamwriter.writerow(["K = " + str(indexK), "P = " + str(indexP), w, "Micro = " + str(fscore_micro), 
                                "Macro = " + str(fscore_macro), "Precision = " + str(prec), "Recall = " + str(rec)])

def guardar_modelo(modelo, file):

    import pickle 
    
    nombreModel = f"models/TopModelKNN-{file}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.sav" 
    saved_model = pickle.dump(modelo, open(nombreModel,'wb'))
    print('Modelo guardado correctamente empleando Pickle')


if __name__ == "__main__":

    param = pedir_param()

    try:
        datos = load_data(param[1])
    except FileNotFoundError:
        print("Fichero de datos no encontrado, asegurate de escribir bien la ruta")
        exit()

    file = param[1].split("data/")[1].split(".csv")[0]

    try:
        prep = load_json(param[5])
    except FileNotFoundError:
        print("Fichero de preproceso no encontrado, asegurate de escribir bien la ruta")
        exit()   

    

    if param[0] == 0:

        top_model = trainKNN(param[2], param[3], param[4], datos, file, prep)

    elif param[0] == 1:
        #TODO:
        exit

    guardar_modelo(top_model, file)


