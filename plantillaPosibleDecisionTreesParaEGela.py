# -*- coding: utf-8 -*-
"""
Script para la implementación del algoritmo de clasificación
"""

import random
import sys
import signal
import argparse
import pandas as pd
import numpy as np
import string
import pickle
import time
import json
import csv
import os
from colorama import Fore
# Sklearn
from sklearn.calibration import LabelEncoder
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, Normalizer, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
# Nltk
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
# Imblearn
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from tqdm import tqdm


#-----------------------------------------------------------------------------------------------------------------------------------------------------


# Obtencion de los materiales(ficheros, datos...)

def signal_handler(sig, frame):
    """
    Función para manejar la señal SIGINT (Ctrl+C)
    :param sig: Señal
    :param frame: Frame
    """
    print("\nSaliendo del programa...")
    sys.exit(0)

def parse_args():
    """
    Función para parsear los argumentos de entrada
    """
    parse = argparse.ArgumentParser(description="Practica de algoritmos de clasificación de datos.")
    parse.add_argument("-m", "--mode", help="Modo de ejecución (train o test)", required=True)
    parse.add_argument("-f", "--file", help="Fichero csv (/Path_to_file)", required=True)
    parse.add_argument("-p", "--preprocess", help="Fichero json (/Path_to_file)", required=True)
    parse.add_argument("-a", "--algorithm", help="Algoritmo a ejecutar (kNN, DT o RF)", required=True)
    parse.add_argument("-c", "--cpu", help="Número de CPUs a utilizar [-1 para usar todos]", required=False, default=-1, type=int)
    parse.add_argument("-v", "--verbose", help="Muestra las metricas por la terminal", required=False, default=False, action="store_true")
    parse.add_argument("--debug", help="Modo debug [Muestra informacion extra del preprocesado y almacena el resultado del mismo en un .csv]", required=False, default=False, action="store_true")
    # Parseamos los argumentos
    args = parse.parse_args()

    # Leemos los parametros del JSON
    with open(args.preprocess) as json_file:
        config = json.load(json_file)
    
        # Juntamos todo en una variable
    for key, value in config.items():
        setattr(args, key, value)
    
    # Parseamos los argumentos
    return args
    
def load_data(file):
    """
    Función para cargar los datos de un fichero csv
    :param file: Fichero csv
    :return: Datos del fichero
    """
    try:
        data = pd.read_csv(file, encoding='utf-8')
        print(Fore.GREEN+"Datos cargados con éxito"+Fore.RESET)
        return data
    except Exception as e:
        print(Fore.RED+"Error al cargar los datos"+Fore.RESET)
        print(e)
        sys.exit(1)


#-----------------------------------------------------------------------------------------------------------------------------------------------------


# Funciones para calcular métricas

#realiza el calculo del fscore(equlibrio entre la precision y el recall)/ micro() y macro()
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


#calcula la matriz de confusion comparando tanto los datos de test como los datos de la prediccion 
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


#calcula la precision y el recall
def calculate_prec_rec(y_test, y_pred):
    """
    opciones de average:
        micro:calcula la global(no por calcula por apartados)
        macro: calcula cada sector y saca el promedio
        weighted: lo mismo que macro pero tiene efecto la cantidad de unidades en cada clase
        sample:calcula cada muestra y el promedio
        none: array con la precision de cada muestra
    """
 
    from sklearn.metrics import precision_score, recall_score
    return precision_score(y_test, y_pred, average="weighted"), recall_score(y_test, y_pred, average="weighted")


#-----------------------------------------------------------------------------------------------------------------------------------------------------


# Funcion auxiliar


def select_features():

    """
    Separa las características del conjunto de datos en características numéricas, de texto y categóricas.

    Returns:
        numerical_feature (DataFrame): DataFrame que contiene las características numéricas.
        text_feature (DataFrame): DataFrame que contiene las características de texto.
        categorical_feature (DataFrame): DataFrame que contiene las características categóricas.
    """
    try:
        target = args.preprocessing["target"]
        # Numerical features
        numerical_feature = data.select_dtypes(include=['int64', 'float64']) # Columnas numéricas
        if target in numerical_feature.columns:
            numerical_feature = numerical_feature.drop(columns=[target])
        # Categorical features
        categorical_feature = data.select_dtypes(include='object')
        categorical_feature = categorical_feature.loc[:, categorical_feature.nunique() <= int(args.preprocessing["unique_category_threshold"])]
        
        # Text features
        text_feature = data.select_dtypes(include='object').drop(columns=categorical_feature.columns)
        print(Fore.GREEN+"Datos separados con éxito"+Fore.RESET)
        
        if args.debug:
            print(Fore.MAGENTA+"> Columnas numéricas:\n"+Fore.RESET, numerical_feature.columns)
            print(Fore.MAGENTA+"> Columnas de texto:\n"+Fore.RESET, text_feature.columns)
            print(Fore.MAGENTA+"> Columnas categóricas:\n"+Fore.RESET, categorical_feature.columns)
        return numerical_feature, text_feature, categorical_feature
    except Exception as e:
        print(Fore.RED+"Error al separar los datos"+Fore.RESET)
        print(e)
        sys.exit(1)


#-----------------------------------------------------------------------------------------------------------------------------------------------------


# Funciones para preprocesar los datos


def preprocesar_datos():
    """
    Función para preprocesar los datos
        1. Separamos los datos por tipos (Categoriales, numéricos y textos)
        2. Pasar los datos de categoriales a numéricos 
        3. Tratamos missing values (Eliminar y imputar)
        4. Reescalamos los datos datos (MinMax, Normalizer, MaxAbsScaler)
        TODO 5. Simplificamos el texto (Normalizar, eliminar stopwords, stemming y ordenar alfabéticamente)
        6. Tratamos el texto (TF-IDF, BOW)
        7. Realizamos Oversampling o Undersampling
        8. Borrar columnas no necesarias
    :param data: Datos a preprocesar
    :return: Datos preprocesados y divididos en train y test
    """
        # Separamos los datos por tipos
    numerical_feature, text_feature, categorical_feature = select_features()

        # Simplificamos el texto
    simplify_text(text_feature)

        # Tratamos el texto
    process_text(text_feature)

        # Pasar los datos a categoriales a numéricos
    cat2num(categorical_feature)

        #Outliers
    outliers(numerical_feature)

        # Tratamos missing values
    process_missing_values(numerical_feature, categorical_feature)

        # Reescalamos los datos numéricos
    reescaler(numerical_feature)

        # Realizamos Oversampling o Undersampling
    over_under_sampling()

    #drop_features()

#-----------------------------------------------------------------------------

def process_missing_values(numerical_feature, categorical_feature):
    """
    Procesa los valores faltantes en los datos según la estrategia especificada en los argumentos.

    Args:
        numerical_feature (DataFrame): El DataFrame que contiene las características numéricas.
        categorical_feature (DataFrame): El DataFrame que contiene las características categóricas.

    Returns:
        None

    Raises:
        None
    """
    nf = numerical_feature
    cf = categorical_feature

    #Tratamiento de missing values
    mv = args.preprocessing["missing_values"]
    ist = args.preprocessing["impute_strategy"]

    global data

    try:
        if mv == "drop":
            data = data.dropna()
        elif mv == "impute":
            if ist == "mean":
                for feature in nf:
                    data[feature] = data[feature].fillna(data[feature].mean())
            if ist == "median":
                for feature in nf:
                    data[feature] = data[feature].fillna(data[feature].median())
            if ist == "mode":
                for feature in nf:
                    data[feature] = data[feature].fillna(data[feature].mode())

        for feature in cf:
            data[feature] = data[feature].fillna(data[feature].mode())

        print(Fore.GREEN+f"Missing values tratados correctamente con la accion: {mv} y estrategia: {ist}"+Fore.RESET)
    
    except Exception as e:
        print(Fore.RED+"Error al tratar missing values"+Fore.RESET)
        print(e)

def reescaler(numerical_feature):
    """
    Rescala las características numéricas en el conjunto de datos utilizando diferentes métodos de escala.

    Args:
        numerical_feature (DataFrame): El dataframe que contiene las características numéricas.

    Returns:
        None

    Raises:
        Exception: Si hay un error al reescalar los datos.

    """
    ##Reescalado
    f = numerical_feature

    e = args.preprocessing["scaling"]

    global data

    try:
        if e == "min-max":
            for feature in f:
                min_value = data[feature].min()
                max_value = data[feature].max()
                
                if min_value != max_value:
                    data[feature] = (data[feature] - min_value) / (max_value - min_value)
                else:
                    print(f"Atributo {feature} no se ha escalado porque min y max son iguales.")

        elif e == "avgstd":
            for feature in f:
                avg = data[feature].mean()
                std = data[feature].std()
                
                if std != 0:
                    data[feature] = (data[feature] - avg) / std
                else:
                    print(f"Atributo {feature} no se ha escalado porque su desviación estándar es 0.")
        
        elif e == "entreMax":
            for feature in f:
                max = data[feature].max()
                if max != 0:
                    data[feature] = data[feature]/max
                else:
                    print(f"Atributo {feature} no se ha escalado porque su maximo estándar es 0.")

        print(Fore.GREEN+f"Reescalado realizado correctamente con: {e}"+Fore.RESET)
    
    except Exception as e:
        print(Fore.RED+"Error al realizar el reescalado"+Fore.RESET)
        print(e)

def cat2num(categorical_feature):
    """
    Convierte las características categóricas en características numéricas utilizando la codificación de etiquetas.

    Parámetros:
    categorical_feature (DataFrame): El DataFrame que contiene las características categóricas a convertir.

    """
    #Pasar las features categoriales a numericas
    global data

    try:
        encoder = LabelEncoder()
        cf = categorical_feature

        if args.preprocessing["targ_categorial"]== "True" :

            cf += args.preprocessing["target"]

        for feature in cf:
            data[feature] = encoder.fit_transform(data[feature])

        print(Fore.GREEN+"Columnas categoriales numerizadas correctamente"+Fore.RESET)

    except Exception as e:
        print(Fore.RED+"Error al numerizar las columnas categoriales"+Fore.RESET)
        print(e)

def simplify_text(text_feature):
    """
    Función que simplifica el texto de una columna dada en un DataFrame. lower,stemmer, tokenizer, stopwords del NLTK....
    
    Parámetros:
    - text_feature: DataFrame - El DataFrame que contiene la columna de texto a simplificar.
    
    Retorna:
    None
    """
    global data

    # Inicializar el stemmer y las stopwords
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))

    # Procesar cada columna de texto
    for col in text_feature:
        data[col] = text_feature[col].astype(str).str.lower()  # Convertir a minúsculas
        data[col] = text_feature[col].apply(lambda x: ' '.join([
            stemmer.stem(word) for word in word_tokenize(x)
            if word.isalpha() and word not in stop_words
        ]))

 #TODO aqui lo que sea preciso en caso de tener texto

def process_text(text_feature):
    """
    Procesa las características de texto utilizando técnicas de vectorización como TF-IDF o BOW.

    Parámetros:
    text_feature (pandas.DataFrame): Un DataFrame que contiene las características de texto a procesar.

    """
    global data
    try:
        if text_feature.columns.size > 0:
            if args.preprocessing["text_process"] == "tf-idf":               
               tfidf_vectorizer = TfidfVectorizer()
               text_data = data[text_feature.columns].apply(lambda x: ' '.join(x.astype(str)), axis=1)
               tfidf_matrix = tfidf_vectorizer.fit_transform(text_data)
               text_features_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
               data = pd.concat([data, text_features_df], axis=1)
               data.drop(text_feature.columns, axis=1, inplace=True)
               print(Fore.GREEN+"Texto tratado con éxito usando TF-IDF"+Fore.RESET)
            elif args.preprocessing["text_process"] == "BOW":
                bow_vecotirizer = CountVectorizer()
                text_data = data[text_feature.columns].apply(lambda x: ' '.join(x.astype(str)), axis=1)
                bow_matrix = bow_vecotirizer.fit_transform(text_data)
                text_features_df = pd.DataFrame(bow_matrix.toarray(), columns=bow_vecotirizer.get_feature_names_out())
                data = pd.concat([data, text_features_df], axis=1)
                print(Fore.GREEN+"Texto tratado con éxito usando BOW"+Fore.RESET)
            else:
                print(Fore.YELLOW+"No se están tratando los textos"+Fore.RESET)
        else:
            print(Fore.YELLOW+"No se han encontrado columnas de texto a procesar"+Fore.RESET)
    except Exception as e:
        print(Fore.RED+"Error al tratar el texto"+Fore.RESET)
        print(e)
        sys.exit(1)

def over_under_sampling():
    """
    Realiza oversampling o undersampling en los datos según la estrategia especificada en args.preprocessing["sampling"].
    
    Args:
        None
    
    Returns:
        None
    
    Raises:
        Exception: Si ocurre algún error al realizar el oversampling o undersampling.
    """
    global data

    columna_objetivo = args.preprocessing["target"]

    porcentaje = float(args.preprocessing["porcentaje"])

    atrib = data.drop(columns= [columna_objetivo])
    targ = data[columna_objetivo]


    if args.preprocessing["sampling"]== "undersampling":
        sampler = RandomUnderSampler(random_state=42)#, sampling_strategy=porcentaje)
    elif args.preprocessing["sampling"]== "oversampling": #copia los datos de la clase minoritaria
        sampler = RandomOverSampler(random_state=42)#, sampling_strategy=porcentaje)
    elif args.preprocessing["sampling"]== "smote": #crea datos sinteticos de la clase minoritaria
        sampler = SMOTE(random_state=42)#, sampling_strategy=porcentaje)

    variable_balanceado, targ_balanceado = sampler.fit_resample(atrib, targ)
    variable_balanceado_df = pd.DataFrame(variable_balanceado, columns=atrib.columns)
    targ_balanceado_df = pd.DataFrame(targ_balanceado, columns=[columna_objetivo])
    datos_balanceados = pd.concat([variable_balanceado_df, targ_balanceado_df], axis=1)

    data = datos_balanceados
  
def drop_features():
    """
    Elimina las columnas especificadas del conjunto de datos.

    Parámetros:
    features (list): Lista de nombres de columnas a eliminar.

    """
    global data
    try:
        data = data.drop(columns=args.preprocessing["drop_features"])
        print(Fore.GREEN+"Columnas eliminadas con éxito"+Fore.RESET)
    except Exception as e:
        print(Fore.RED+"Error al eliminar columnas"+Fore.RESET)
        print(e)
        sys.exit(1)

def outliers(numerical_feature):

    #Outliers
    nf = numerical_feature

    o = args.preprocessing["outliers"]
    ost = args.preprocessing["round_strategy"]

    global data

    try:
        for feature in nf:
            if ost == "quantile":
                Q1 = data[feature].quantile(0.25)
                Q3 = data[feature].quantile(0.75)
                IQR = Q3 - Q1

                limite_inferior = Q1 - 1.5 * IQR
                limite_superior = Q3 + 1.5 * IQR
            
            elif ost == "std":
                mean = data[feature].mean()
                std = data[feature].std()

                limite_inferior = mean - 3 * std
                limite_superior = mean + 3 * std

                data[feature] = data[feature].astype(float) #No es necesario pero numpy lo recomienda para avitar futuros problemas

            if o == "drop":
                data = data[((data[feature]>=limite_inferior) & (data[feature]<=limite_superior)) 
                            | (data[feature].isna())]

            elif o == "round":
                data.loc[data[feature]<=limite_inferior, feature] = limite_inferior
                data.loc[data[feature]>=limite_superior, feature] = limite_superior

        print(Fore.GREEN+f"Outliers tratados correctamente con la accion: {o} y estrategia: {ost}"+Fore.RESET)
    
    except Exception as e:
        print(Fore.RED+"Error al tratar outliers"+Fore.RESET)
        print(e)


#----------------------------------------------------------------------------------------------------------------------------------------------------


# Funciones para entrenar un modelo

def divide_data():
    """
    Función que divide los datos en conjuntos de entrenamiento y desarrollo.

    Parámetros:
    - data: DataFrame que contiene los datos.
    - args: Objeto que contiene los argumentos necesarios para la división de datos.

    Retorna:
    - x_train: DataFrame con las características de entrenamiento.
    - x_dev: DataFrame con las características de desarrollo.
    - y_train: Serie con las etiquetas de entrenamiento.
    - y_dev: Serie con las etiquetas de desarrollo.
    """
    # Seleccionamos las características y la clase
    y = data[args.preprocessing["target"]] # Última columna
    x = data.drop(columns=args.preprocessing["target"])# Todas las columnas menos la última

    # Dividimos los datos en entrenamiento y test
    from sklearn.model_selection import train_test_split
    np.random.seed(42)  # Set a random seed for reproducibility
    x_train, x_dev, y_train, y_dev = train_test_split(x.values, y.values, test_size= 1 - float(args.train["train_size"]), stratify=y)

    x_dev, x_test, y_dev, y_test = train_test_split(x_dev, y_dev, test_size=float(args.train["test_size"]), stratify=y_dev)
    
    
    x_test = pd.DataFrame(x_test, columns=x.columns)
    y_test = pd.DataFrame(y_test, columns=[args.preprocessing["target"]])
    datos_test = pd.concat([x_test, y_test], axis=1)
    datos_test.to_csv(f"output/{args.file.split('/')[-1].split('.csv')[0]}-test.csv", index = False)

    return x_train, x_dev, y_train, y_dev
 
 
def save_model(gs, x_train, y_train):
    """
    Guarda el modelo y los resultados de la búsqueda de hiperparámetros en archivos.

    Parámetros:
    - gs: objeto GridSearchCV, el cual contiene el modelo y los resultados de la búsqueda de hiperparámetros.

    Excepciones:
    - Exception: Si ocurre algún error al guardar el modelo.

    """

    results = gs.cv_results_
    df_results = pd.DataFrame(results)
    hp = [col for col in df_results.columns if col.startswith('param_')]
    scores = [f'mean_test_{metric}' for metric in args.metrics['evaluation']]
    df_results['Parámetros'] = df_results[hp].apply(lambda row: str(row.to_dict()), axis=1)
    final_results = df_results[['Parámetros'] + scores].copy()
    final_results.to_csv(f"output/{args.file.split('/')[-1].split('.csv')[0]}-informe.csv", index=False, float_format="%.3f")
    best_model_metric = args.metrics['best_model']
    num = int(args.metrics['num_modelos'])
    best_models_indices = df_results.sort_values(by=f'mean_test_{best_model_metric}', ascending=False).head(num).index
    
    try:
        for i, idx in enumerate(best_models_indices):
            best_model_params = gs.cv_results_['params'][idx]
            if args.algorithm=="kNN":
                model = KNeighborsClassifier(**best_model_params) 
            elif args.algorithm == "decision_tree":  
                model = DecisionTreeClassifier(**best_model_params)
            elif args.algorithm == "random_forest":
                model = RandomForestClassifier(**best_model_params) 
            model.fit(x_train, y_train)
            
            model_filename = f"output/{args.file.split('/')[-1].split('.csv')[0]}-modelo-{args.algorithm}-top{i+1}.pkl"
            with open(model_filename, 'wb') as file:
                pickle.dump(model, file)
                print(Fore.CYAN+"Modelo guardado con éxito"+Fore.RESET)

    except Exception as e:
        print(Fore.RED+"Error al guardar el modelo"+Fore.RESET)
        print(e)

def mostrar_resultados(gs, x_dev, y_dev):
    """
    Muestra los resultados del clasificador.

    Parámetros:
    - gs: objeto GridSearchCV, el clasificador con la búsqueda de hiperparámetros.
    - x_dev: array-like, las características del conjunto de desarrollo.
    - y_dev: array-like, las etiquetas del conjunto de desarrollo.

    Imprime en la consola los siguientes resultados:
    - Mejores parámetros encontrados por la búsqueda de hiperparámetros.
    - Mejor puntuación obtenida por el clasificador.
    - F1-score micro del clasificador en el conjunto de desarrollo.
    - F1-score macro del clasificador en el conjunto de desarrollo.
    - Informe de clasificación del clasificador en el conjunto de desarrollo.
    - Matriz de confusión del clasificador en el conjunto de desarrollo.
    """
    if args.verbose:
        print(Fore.MAGENTA+"> Mejores parametros:\n"+Fore.RESET, gs.best_params_)
        print(Fore.MAGENTA+"> Mejor puntuacion:\n"+Fore.RESET, gs.best_score_)
        print(Fore.MAGENTA+"> F1-score micro:\n"+Fore.RESET, calculate_fscore(y_dev, gs.predict(x_dev))[0])
        print(Fore.MAGENTA+"> F1-score macro:\n"+Fore.RESET, calculate_fscore(y_dev, gs.predict(x_dev))[1])
        print(Fore.MAGENTA+"> Matriz de confusión:\n"+Fore.RESET, calculate_confusion_matrix(y_dev, gs.predict(x_dev)))


#-----------------------------------------------------------------------------------------------------------------------------------------------------


#Entrenamiento de los 3 algoritmos


def kNN():
    """
    Función para implementar el algoritmo kNN.
    Hace un barrido de hiperparametros para encontrar los parametros optimos

    :param data: Conjunto de datos para realizar la clasificación.
    :type data: pandas.DataFrame
    :return: Tupla con la clasificación de los datos.
    :rtype: tuple
    """
    # Dividimos los datos en entrenamiento y dev
    x_train, x_dev, y_train, y_dev = divide_data()

    hp = {
        'n_neighbors': range(int(args.kNN["k"]), int(args.kNN["K"])+1),
        'weights': ["uniform", "distance"],
        'p': args.kNN["p"]
        }
    
    # Hacemos un barrido de hiperparametros

    with tqdm(total=100, desc='Procesando kNN', unit='iter', leave=True) as pbar:
        gs = GridSearchCV(KNeighborsClassifier(), hp, cv=5, n_jobs=args.cpu, scoring=args.metrics["evaluation"], refit=args.metrics["best_model"])
        start_time = time.time()
        gs.fit(x_train, y_train)
        end_time = time.time()
        for i in range(100):
            time.sleep(random.uniform(0.06, 0.15))  # Esperamos un tiempo aleatorio
            pbar.update(random.random()*2)  # Actualizamos la barra con un valor aleatorio
        pbar.n = 100
        pbar.last_print_n = 100
        pbar.update(0)
    execution_time = end_time - start_time
    print("Tiempo de ejecución:"+Fore.MAGENTA, execution_time,Fore.RESET+ "segundos")
    
    # Mostramos los resultados
    mostrar_resultados(gs, x_dev, y_dev)
    
    # Guardamos el modelo utilizando pickle
    save_model(gs, x_train, y_train)

def decision_tree():
    """
    Función para implementar el algoritmo de árbol de decisión.

    :param data: Conjunto de datos para realizar la clasificación.
    :type data: pandas.DataFrame
    :return: Tupla con la clasificación de los datos.
    :rtype: tuple
    """
    # Dividimos los datos en entrenamiento y dev
    x_train, x_dev, y_train, y_dev = divide_data()
    
    # Hacemos un barrido de hiperparametros
    hp = {
        'criterion': ['gini', 'entropy'],
        'splitter': ['best', 'random'],
        'max_depth': range(int(args.DT["min_depth"]), int(args.DT["max_depth"])+1),
        'min_samples_leaf': range(args.DT["intervalo_sample_per_leaf"][0], args.DT["intervalo_sample_per_leaf"][-1]+1)
        }

    with tqdm(total=100, desc='Procesando DT', unit='iter', leave=True) as pbar:
        gs = GridSearchCV(DecisionTreeClassifier(), hp, cv=5, n_jobs=args.cpu, scoring=args.metrics["evaluation"], refit=args.metrics["best_model"])
        start_time = time.time()
        gs.fit(x_train, y_train)
        end_time = time.time()
        for i in range(100):
            time.sleep(random.uniform(0.06, 0.15))  # Esperamos un tiempo aleatorio
            pbar.update(random.random()*2)  # Actualizamos la barra con un valor aleatorio
        pbar.n = 100
        pbar.last_print_n = 100
        pbar.update(0)
    execution_time = end_time - start_time
    print("Tiempo de ejecución:"+Fore.MAGENTA, execution_time,Fore.RESET+ "segundos")
    
    # Mostramos los resultados
    mostrar_resultados(gs, x_dev, y_dev)
    
    # Guardamos el modelo utilizando pickle
    save_model(gs, x_train,  y_train)
    
def random_forest():
    """
    Función que entrena un modelo de Random Forest utilizando GridSearchCV para encontrar los mejores hiperparámetros.
    Divide los datos en entrenamiento y desarrollo, realiza la búsqueda de hiperparámetros, guarda el modelo entrenado
    utilizando pickle y muestra los resultados utilizando los datos de desarrollo.

    Parámetros:
        Ninguno

    Retorna:
        Ninguno
    """
    
    # Dividimos los datos en entrenamiento y dev
    x_train, x_dev, y_train, y_dev = divide_data()
    
    # Hacemos un barrido de hiperparametros

    hp = {
        'criterion': ['gini', 'entropy'],
        'max_depth': range(int(args.RF["min_depth"]), int(args.RF["max_depth"])+1),
        'n_estimators': range(int(args.RF["n_estimators_min"]), int(args.RF["n_estimators_max"])+1),
        'min_samples_leaf': range(args.RF["intervalo_sample_per_leaf"][0], args.RF["intervalo_sample_per_leaf"][-1]+1)
        }

    with tqdm(total=100, desc='Procesando DT', unit='iter', leave=True) as pbar:
        gs = GridSearchCV(RandomForestClassifier(), hp, cv=5, n_jobs=args.cpu, scoring=args.metrics["evaluation"], refit=args.metrics["best_model"])
        start_time = time.time()
        gs.fit(x_train, y_train)
        end_time = time.time()
        for i in range(100):
            time.sleep(random.uniform(0.06, 0.15))  # Esperamos un tiempo aleatorio
            pbar.update(random.random()*2)  # Actualizamos la barra con un valor aleatorio
        pbar.n = 100
        pbar.last_print_n = 100
        pbar.update(0)
    execution_time = end_time - start_time
    print("Tiempo de ejecución:"+Fore.MAGENTA, execution_time,Fore.RESET+ "segundos")
    
    # Mostramos los resultados
    mostrar_resultados(gs, x_dev, y_dev)
    
    # Guardamos el modelo utilizando pickle
    save_model(gs, x_train,  y_train)



#-----------------------------------------------------------------------------------------------------------------------------------------------------


# Funciones para predecir con un modelo

def load_model():
    """
    Carga el modelo desde el archivo 'output/modelo.pkl' y lo devuelve.

    Returns:
        model: El modelo cargado desde el archivo 'output/modelo.pkl'.

    Raises:
        Exception: Si ocurre un error al cargar el modelo.
    """
    try:
        with open(f"output/{args.test['model_to_test']}.pkl", 'rb') as file:
            model = pickle.load(file)
            print(Fore.GREEN+"Modelo cargado con éxito"+Fore.RESET)
            return model
    except Exception as e:
        print(Fore.RED+"Error al cargar el modelo"+Fore.RESET)
        print(e)
        sys.exit(1)
        
def predict():
    """
    Realiza una predicción utilizando el modelo entrenado y guarda los resultados en un archivo CSV.

    Parámetros:
        Ninguno

    Retorna:
        Ninguno
    """
    global data
    # Predecimos
    atributos = data.drop(columns=args.preprocessing["target"])
    prediction = model.predict(atributos.values)
    
    # Añadimos la prediccion al dataframe data
    data = pd.concat([data, pd.DataFrame(prediction, columns=["prediccion"])], axis=1)

    
    precission , recall = calculate_prec_rec(data[args.preprocessing["target"]], data["prediccion"])
    f_score = calculate_fscore(data[args.preprocessing["target"]], data["prediccion"])
    with open(f"output/{args.file.split('/')[-1].split('-')[0]}-informeDelTest.csv", 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(["Parametros","precission", "recall", "f_score"])
        writer.writerow([model.get_params() , precission, recall, f_score])

  #-----------------------------------------------------------------------------------------------------------------------------------------------------


#-----------------------------------------------------------------------------------------------------------------------------------------------------


# Función principal

if __name__ == "__main__":

    # Fijamos la semilla
    np.random.seed(42)
    print("=== Clasificador ===")

    # Manejamos la señal SIGINT (Ctrl+C)
    signal.signal(signal.SIGINT, signal_handler)

    # Parseamos los argumentos
    args = parse_args()

    # Si la carpeta output no existe la creamos
    print("\n- Creando carpeta output...")
    try:
        os.makedirs('output')
        print(Fore.GREEN+"Carpeta output creada con éxito"+Fore.RESET)
    except FileExistsError:
        print(Fore.GREEN+"La carpeta output ya existe"+Fore.RESET)
    except Exception as e:
        print(Fore.RED+"Error al crear la carpeta output"+Fore.RESET)
        print(e)
        sys.exit(1)

    # Cargamos los datos
    print("\n- Cargando datos...")
    data = load_data(args.file)

    # Descargamos los recursos necesarios de nltk
    print("\n- Descargando diccionarios...")
    nltk.download('stopwords')
    nltk.download('punkt_tab')
    nltk.download('wordnet')

    # Preprocesamos los datos
    if args.mode =="train":
        print("\n- Preprocesando datos...")
        preprocesar_datos()

    if args.debug:
        try:
            print("\n- Guardando datos preprocesados...")
            data.to_csv(f'output/{args.file.split("/")[-1].split(".csv")[0]}-processed.csv', index=False)
            print(Fore.GREEN+"Datos preprocesados guardados con éxito"+Fore.RESET)
        except Exception as e:
            print(Fore.RED+"Error al guardar los datos preprocesados"+Fore.RESET)
            print(e)

    if args.mode == "train":
        # Ejecutamos el algoritmo seleccionado
        print("\n- Ejecutando algoritmo...")
        if args.algorithm == "kNN":
            try:
                kNN()
                print(Fore.GREEN+"Algoritmo kNN ejecutado con éxito"+Fore.RESET)
                sys.exit(0)
            except Exception as e:
                print(e)
        elif args.algorithm == "decision_tree":
            try:
                decision_tree()
                print(Fore.GREEN+"Algoritmo árbol de decisión ejecutado con éxito"+Fore.RESET)
                sys.exit(0)
            except Exception as e:
                print(e)
        elif args.algorithm == "random_forest":
            try:
                random_forest()
                print(Fore.GREEN+"Algoritmo random forest ejecutado con éxito"+Fore.RESET)
                sys.exit(0)
            except Exception as e:
                print(e)
        else:
            print(Fore.RED+"Algoritmo no soportado"+Fore.RESET)
            sys.exit(1)

    elif args.mode == "test":

        # Cargamos el modelo
        print("\n- Cargando modelo...")
        model = load_model()

        # Predecimos
        print("\n- Prediciendo...")
        try:
            predict()
            print(Fore.GREEN+"Predicción realizada con éxito"+Fore.RESET)
            # Guardamos el dataframe con la prediccion
            data.to_csv(f'output/{args.file.split("/")[-1].split(".csv")[0]}-prediction.csv', index=False)

            print(Fore.GREEN+"Predicción guardada con éxito"+Fore.RESET)
            sys.exit(0)
        except Exception as e:
            print(e)
            sys.exit(1)
    else:
        print(Fore.RED+"Modo no soportado"+Fore.RESET)
        sys.exit(1)
