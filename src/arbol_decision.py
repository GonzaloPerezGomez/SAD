import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

#---------------------------------------------------------------------------------------------------------------------------------#
#entrenamiento knn del modelo
def trainDT(k, K, p, datos, file, conf):

    return None

def gen_informe_DT():

    return None
    

def decisition_tree(data: pd.DataFrame, criterion: str, split_strat: str, max_depth : int, min_sample_per_leaf):

    # Seleccionamos las características y la clase
    X = data.iloc[:, :-1].values # Todas las columnas menos la última
    y = data.iloc[:, -1].values # Última columna
    
    # Dividimos los datos en entrenamiento y test
    from sklearn.model_selection import train_test_split
    np.random.seed(42)  # Set a random seed for reproducibility
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    
    classifier = DecisionTreeClassifier(criterion= criterion, splitter= split_strat, 
                                        max_depth= max_depth, min_samples_leaf= min_sample_per_leaf)
   
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    
    return y_test, y_pred, classifier