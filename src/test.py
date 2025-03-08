import pandas as pd
import os
import pickle
import preprocesado
#TODO: cambiarlo para poder elegir con que algoritmo entrenaste el modelo
def params():

    algo = int(input("Sobre que algoritmo entrenaste el modelo(KNN = 0, DT = 1, RF=2, NB=3): "))

    if algo == 0:
        algo = "KNN"
    elif algo == 1:
        algo = "DT"
    elif algo == 2:
        algo = "RF"
    elif algo == 3:
        algo = "NB"

    print("Sobre que dataset entrenaste el modelo: ")

    modelo = -1

    while modelo <= 0 or modelo > i:
        i = 0
        for m in os.listdir(f"models/{algo}"):
            i += 1
            print(f"{i} - {m.split('-')[1].split('.sav')[0]}")
            
        modelo = int(input())

        if modelo <= 0 or modelo > i:
            print("Valor no valido, vuelve a intentarlo")
    
    modelo = os.listdir(f"models/{algo}")[modelo-1]
    modelo = pickle.load(open(f"models/{algo}/{modelo}", 'rb'))

    data_set = input("Sobre que dataset vas a querer testear: ")
    
    try:
        data_set = pd.read_csv(data_set)
    except FileNotFoundError:
        print("Fichero de dataset no encontrado")

    conf = input("Indique el fichero json para el preprocesado(dejar en blanco si no requiere preprocesado): ")
   
    try:
        conf = pd.read_json(conf)
    except FileNotFoundError:
        print("Fichero de preprocesado no encontrado")

    return data_set, conf, modelo, algo

def predecir(data_prep, modelo):

    print(modelo.predict(data_prep.values))


if __name__ == "__main__":

    data, prep, modelo, algo = params()

    if algo == "KNN":
        predecir(preprocesado.preprocesadoKNN(data, prep), modelo)
    elif algo == "DT":
        pass
    elif algo == "RF":
        pass
    elif algo == "NB":
        pass