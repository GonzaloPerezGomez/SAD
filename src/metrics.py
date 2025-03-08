

#------------------------------------------------------------------------------------------------------------------------------#

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


#------------------------------------------------------------------------------------------------------------------------------#

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


#------------------------------------------------------------------------------------------------------------------------------#

#mirar interior(info de average)

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
