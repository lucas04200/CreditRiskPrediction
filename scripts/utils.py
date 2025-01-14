import numpy as np 
np.set_printoptions(threshold=10000,suppress=True) 
import pandas as pd 
import warnings 
import matplotlib.pyplot as plt 
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import recall_score, make_scorer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_selection import SelectFromModel
import pickle
warnings.filterwarnings('ignore')


# Models
clfs={
    'CART' : DecisionTreeClassifier(criterion='gini',random_state=1),
    'ID3' : DecisionTreeClassifier(criterion='entropy',random_state=1),
    'Stump' : DecisionTreeClassifier(criterion='gini',max_depth=1,random_state=1),
    'KNN': KNeighborsClassifier(n_neighbors=5,n_jobs=-1),
    'Bag': BaggingClassifier(estimator=DecisionTreeClassifier(random_state=1),
                             n_estimators=200,random_state=1,n_jobs=-1),
    'Ad': AdaBoostClassifier(n_estimators=200,random_state=1),
    'RF':RandomForestClassifier(n_estimators=200,random_state=1,n_jobs=-1),
    'ExtraTree':ExtraTreesClassifier(n_estimators=200,random_state=1,n_jobs=-1),
    'MLP_1':MLPClassifier(hidden_layer_sizes=(10),random_state=1),
    'MLP_2':MLPClassifier(hidden_layer_sizes=(30,10),random_state=1),
}

# Fonction custom loss
def my_custom_loss_func(Ytrue, Ypred):
    return (recall_score(Ytrue,Ypred)+recall_score(Ytrue,Ypred,pos_label=0))/2

# Fonction de scoring
moyenne_recall = make_scorer(my_custom_loss_func, greater_is_better=True)

# Fonction de run des modèles
def run_classifier(X, y, clfs):
    kf = KFold(n_splits=10, shuffle=True, random_state=0) 
    best_model = None
    best_score = -np.inf
    
    for name, clf in clfs.items():  # Parcourir les classifieurs
        # CrossValisation sur le random forest, et le scoring est notre score personnalisé 
        cv_acc = cross_val_score(clf, X, y, cv=kf, scoring=moyenne_recall)
        # Moyenne de notre score    
        mean_score = np.mean(cv_acc)
        # ecart-types de notre scoring 
        std_score = np.std(cv_acc)
        # print des score de nos modèles
        print(f"Score for {name} is: {mean_score:.3f} +/- {std_score:.3f}")
        
        # si la moyenne de scoring est supérieur au dernier meilleur score alors on mets à jour le best score par la dernière moyenne obtenue
        if mean_score > best_score:
            best_score = mean_score
            best_model = name
    
    # Best model
    print(f"\nBest model: {best_model} with score: {best_score:.3f}")
    return best_model

# Choix de variables 
def selection_variables(X,Y,features):
    # utilisation du random forest pour obtenir l'importance des features
    clf = RandomForestClassifier(n_estimators=1000,random_state=1)
    clf.fit(X, Y)
    # fonction pour afficher l'importances des features 
    importances=clf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in clf.estimators_],axis=0)
    # Sort des variables d'importances dans l'ordre décroissant 
    sorted_idx = np.argsort(importances)[::-1]
    print(features[sorted_idx])
    # Padding calculation
    padding = np.arange(X.size/len(X)) + 0.5
    plt.barh(padding, importances[sorted_idx],xerr=std[sorted_idx], align='center')
    # Plot des variables importantes
    plt.yticks(padding, features[sorted_idx])
    plt.xlabel("Relative Importance")
    plt.title("Variable Importance")
    plt.show()
    return sorted_idx

# Nombre de variables selection
def nb_variables(X,Y,features, best_model, sorted_idx):
    # Initialisation
    scores = np.zeros(len(features))
    kf = KFold(n_splits=10, shuffle=True, random_state=0)
    # Boucle sur le nombre de variables à sélectionner
    for f in range(0, X.shape[1]):
        cv_moy_recall = cross_val_score(best_model, X[:,sorted_idx[:f+1]], Y, cv=kf, scoring=moyenne_recall, n_jobs=-1)
        scores[f] = np.mean(cv_moy_recall)
    # Visualisation
    plt.plot(scores) 
    plt.xlabel("Nombre de Variables")
    plt.ylabel("Moyenne des rappels")
    plt.title("Evolution de la moyenne des rappels en fonction des variables")
    plt.show()
    return scores

