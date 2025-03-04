import pandas as pd
import os
import pickle
import kNN

def params():

    print("Sobre que dataset entrenaste el modelo: ")

    modelo = -1

    while modelo <= 0 or modelo > i:
        i = 0
        for m in os.listdir("models"):
            i += 1
            print(f"{i} - {m.split('-')[1].split('.sav')[0]}")
            
        modelo = int(input())

        if modelo <= 0 or modelo > i:
            print("Valor no valido, vuelve a intentarlo")
    
    modelo = os.listdir("models")[modelo-1]
    modelo = pickle.load(open("models/" + modelo, 'rb'))

    data_set = input("Sobre que dataset vas a querer testear: ")
    
    try:
        data_set = pd.read_csv(data_set)
    except FileNotFoundError:
        print("Fichero de dataset no encontrado")

    conf = input("Indique el fichero json para el preprocesado(dejar en blanco si no requiere preprocesado)")
   
    try:
        conf = pd.read_json(conf)
    except FileNotFoundError:
        print("Fichero de preprocesado no encontrado")

    return data_set, conf, modelo

def predecir(data_prep, modelo):

    print(modelo.predict(data_prep.values))


if __name__ == "__main__":

    data, prep, modelo = params()

    predecir(kNN.preprocesadoKNN(data, prep), modelo)