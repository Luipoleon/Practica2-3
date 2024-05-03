
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import warnings
from sklearn.exceptions import ConvergenceWarning

# Función para entrenar a los modelos
def train_models(X_train, y_train):
    # Inicializar y entrenar los modelos
    logistic_regression = LogisticRegression(max_iter=1000)
    knn = KNeighborsClassifier(n_neighbors=3) 
    svm = SVC(max_iter=1000)
    naive_bayes = GaussianNB()
    neural_network = MLPClassifier(hidden_layer_sizes=(100, 100, 100), max_iter=1000)
    logistic_regression.fit(X_train, y_train)
    knn.fit(X_train, y_train)
    svm.fit(X_train, y_train)
    naive_bayes.fit(X_train, y_train)
    neural_network.fit(X_train, y_train)
    return logistic_regression, knn, svm, naive_bayes, neural_network

# Función para predecir con los modelos
def predict_models(models, X_test):
    logistic_regression, knn, svm, naive_bayes, neural_network = models
    y_pred_lr = logistic_regression.predict(X_test)
    y_pred_knn = knn.predict(X_test)
    y_pred_svm = svm.predict(X_test)
    y_pred_nb = naive_bayes.predict(X_test)
    y_pred_nn = neural_network.predict(X_test)
    return y_pred_lr, y_pred_knn, y_pred_svm, y_pred_nb, y_pred_nn
# Función para evaluar los modelos

def evaluate_models(y_test, y_pred, binary=False):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    # Calcular binary specificity
    if binary:
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        specificity = tn / (tn + fp)
    else :
        # Calcular specificity multiclass

        # Calcular la matriz de confusión
        cm = confusion_matrix(y_test, y_pred)
        # Calcular la especificidad
        tn = np.diag(cm)
        fp = np.sum(cm, axis=0) - tn
        fn = np.sum(cm, axis=1) - tn
        tp = np.sum(cm) - (tn + fp + fn)
        specificity = tn / (tn + fp)
        specificity = np.mean(specificity)

    return accuracy, precision, recall, f1, specificity


# Función para imprimir los resultados
def print_results(accuracy_lr, precision_lr, recall_lr, f1_lr, y_pred_lr, y_test, specificity_lr=None):
    print('Accuracy:', accuracy_lr)
    print('Precision:', precision_lr)
    print('Recall:', recall_lr)
    print('Confusion Matrix:')
    print(confusion_matrix(y_test, y_pred_lr))
    print('Specificity:', specificity_lr)
    print('F1:', f1_lr)
    print()



if __name__ == "__main__":

    # Avoid max num of iterations warning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    # Cargar los datasets
    swedish_auto_insurance = pd.read_csv('slr06.csv')
    wine_quality = pd.read_csv('winequality-white.csv', sep=';')
    pima_indians_diabetes = pd.read_csv('pima-indians-diabetes.csv', header=None)
    # Separar las características de las etiquetas
    X_swedish = np.array(swedish_auto_insurance['X']).reshape(-1, 1)
    y_swedish = np.array(swedish_auto_insurance['Y'])
    X_wine = np.array(wine_quality[['fixed acidity', 'chlorides', 'free sulfur dioxide',
                            'total sulfur dioxide', 'density', 'pH', 'sulphates', 
                            'alcohol']])
    y_wine = np.array(wine_quality['quality']).reshape(-1, 1)
    X_pima = np.array(pima_indians_diabetes.iloc[:, :-1])
    y_pima = np.array(pima_indians_diabetes.iloc[:, -1])


    # Clasificar los valores de salida del dataset swedish_auto_insurance en tres clases (bajo, medio, alto)
    for i in range(len(y_swedish)):
        if y_swedish[i] < 100:
            y_swedish[i] = 0
        elif y_swedish[i] < 200:
            y_swedish[i] = 1
        else:
            y_swedish[i] = 2

        
    # Dividir los datasets en conjunto de entrenamiento y prueba
    X_train_swedish, X_test_swedish, y_train_swedish, y_test_swedish = train_test_split(X_swedish, y_swedish.ravel(), test_size=0.2, random_state=42)
    X_train_wine, X_test_wine, y_train_wine, y_test_wine = train_test_split(X_wine, y_wine.ravel(), test_size=0.2, random_state=42)
    X_train_pima, X_test_pima, y_train_pima, y_test_pima = train_test_split(X_pima, y_pima.ravel(), test_size=0.2, random_state=42)
    
    for i in range(3):
        if i == 0:
            print('Swedish Auto Insurance Dataset\n')
            X_train = X_train_swedish
            y_train = y_train_swedish
            X_test = X_test_swedish
            y_test = y_test_swedish
        elif i == 1:
            print('Wine Quality Dataset\n')
            X_train = X_train_wine
            y_train = y_train_wine
            X_test = X_test_wine
            y_test = y_test_wine
        else:
            print('Pima Indians Diabetes Dataset\n')
            X_train = X_train_pima
            y_train = y_train_pima
            X_test = X_test_pima
            y_test = y_test_pima
        
        # Entrenar los modelos
        models = train_models(X_train, y_train)
        # Predecir con los modelos
        y_pred_lr, y_pred_knn, y_pred_svm, y_pred_nb, y_pred_nn = predict_models(models, X_test)
        # Evaluar los modelos
        accuracy_lr, precision_lr, recall_lr, f1_lr, specificity_lr = evaluate_models(y_test, y_pred_lr)
        accuracy_knn, precision_knn, recall_knn, f1_knn, specificity_knn = evaluate_models(y_test, y_pred_knn)
        accuracy_svm, precision_svm, recall_svm, f1_svm, specificity_svm = evaluate_models(y_test, y_pred_svm)
        accuracy_nb, precision_nb, recall_nb, f1_nb, specificity_nb = evaluate_models(y_test, y_pred_nb)
        accuracy_nn, precision_nn, recall_nn, f1_nn, specificity_nn = evaluate_models(y_test, y_pred_nn)
       
        # Imprimir los resultados
        print('Logistic Regression')
        print_results(accuracy_lr, precision_lr, recall_lr, f1_lr, y_pred_lr, y_test, specificity_lr)
        print('K-Nearest Neighbors')
        print_results(accuracy_knn, precision_knn, recall_knn, f1_knn, y_pred_knn, y_test, specificity_knn)
        print('Support Vector Machine')
        print_results(accuracy_svm, precision_svm, recall_svm, f1_svm, y_pred_svm, y_test, specificity_svm)
        print('Naive Bayes')
        print_results(accuracy_nb, precision_nb, recall_nb, f1_nb, y_pred_nb, y_test, specificity_nb)
        print('Neural Network')
        print_results(accuracy_nn, precision_nn, recall_nn, f1_nn, y_pred_nn, y_test, specificity_nn)
    
    