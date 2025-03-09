
import csv
import pandas as pd
import kNN
import preprocesado
from datetime import datetime
import tkinter as tk
from tkinter import filedialog


#----------------------------------------------------------------------------------------------------------------------------#

#pide los parametro del algoritmo al usuario que lo este utilizando a parte del json donde se guarde el preprocesado que tenga que realizar
def pedir_param():
    root = tk.Tk()
    root.withdraw()

    file = filedialog.askopenfilename(title="Selecciona un archivo CSV", filetypes=[("Archivos CSV", "*.csv")])
    algoritmo = int(input("Selecciona el algoritmo a usar(KNN = 0, DT = 1, RF=2, NB=3)"))

    if algoritmo == 0:
        while True:
            k = int(input("Numero minimo de vecinos(k): "))
            K = int(input("Numero maximo de vecinos(k): "))
            p = int(input("Numero maximo de p (1,2): "))
            if k <= 0 or K <= 0:
                print("Los numeros de vecinos deben ser positivos")
            elif K < k:
                print("El numero maximo de vecinos no puede ser mayor al numero minimo de vecinos")
            elif p != 1 and p != 2:
                print("El valor de p debe ser 1 o 2")
            else:
                break

        conf = input("Indique el fichero json para el preprocesado(dejar en blanco si no requiere preprocesado)")
        return [algoritmo, file, k, K, p, conf]

    elif algoritmo == 1:
        #TODO: Definir alforitmo de arbol
         while True:
            min_depth = int(input("Minima profundidad: "))
            max_depth = int(input("Maxima profundidad"))
            #Minimo de instancias por hoja
            if min_depth <= 0 or max_depth <= 0:
                print("Las profundidades no pueden ser menores que 1")
            elif max_depth < min_depth:
                print("El numero maximo de profundidad no puede ser mayor al numero minimo de profundidad")
            elif p != 1 and p != 2:
                print("El valor de p debe ser 1 o 2")
            else:
                break

            conf = input("Indique el fichero json para el preprocesado(dejar en blanco si no requiere preprocesado): ")
            exit()
            return [algoritmo, file, k, K, p, conf]

    elif algoritmo == 2:
        #TODO: Definir alforitmo de random forest
        exit()
    
    elif algoritmo ==3:
          #TODO: Definir alforitmo de ____________
        exit()
    else:
        print("Valor no valido")
        exit()


#------------------------------------------------------------------------------------------------------------------------------#

#Carga el file para realizar el preprocesado y la muestra
def load_data(file):
    """
    Función para cargar los datos de un fichero csv
    :param file: Fichero csv
    :return: Datos del fichero
    """
    data = pd.read_csv(file)
    return data

#------------------------------------------------------------------------------------------------------------------------------#

#cargar el json que dira como realizar el preprocesado
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

#--------------------------------------------------------------------------------------------------------------------------------#

#guarda la informacion obtenida en el entrenamiento _____ en un fichero .sav que se encuentra en la carpeta informes
def guardar_modelo(modelo, file, algo):
    import pickle 
    
    nombreModel = f"models/{algo}/TopModel{algo}-{file}.sav" 
    saved_model = pickle.dump(modelo, open(nombreModel,'wb'))
    print('Modelo guardado correctamente empleando Pickle')


#--------------------------------------------------------------------------------------------------------------------------------#

#clase principal y desde la que se llamaran a los procesos
if __name__ == "__main__":
    #pedir parametros(1ºproceso)
    param = pedir_param()
    try:
        #cargado de datos del file 
        datos = load_data(param[1])
    except FileNotFoundError:
        print("Fichero de datos no encontrado, asegurate de escribir bien la ruta")
        exit()

    #obtemos el nombre del fichero
    file = param[1].split("data/")[1].split(".csv")[0]

    try:
        #cargamoes el json
        prep = load_json(param[5])
    except FileNotFoundError:
        print("Fichero de preproceso no encontrado, asegurate de escribir bien la ruta")
        exit()   

    #si algoritmo es KNN
    if param[0] == 0:
        #entrenamos al modelo
        #   param[2]--> k   /  param[3]--> K   /  param[4]--> p
        top_model = kNN.trainKNN(param[2], param[3], param[4], datos, file, prep)
        algo = "KNN"


    #si algoritmo es Arbol Binario
    elif param[0] == 1:
        algo = "DT"
        #TODO:
        exit
    
    elif param[0] == 2:
        algo = "RF"
        #TODO:
        exit

    guardar_modelo(top_model, file, algo)


