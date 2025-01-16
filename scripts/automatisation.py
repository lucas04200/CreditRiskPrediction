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


# Model dic
def get_clfs():
    return {
        'CART': DecisionTreeClassifier(criterion='gini', random_state=1),
        'ID3': DecisionTreeClassifier(criterion='entropy', random_state=1),
        'Stump': DecisionTreeClassifier(criterion='gini', max_depth=1, random_state=1),
        'KNN': KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        'Bag': BaggingClassifier(estimator=DecisionTreeClassifier(random_state=1), n_estimators=200, random_state=1, n_jobs=-1),
        'Ad': AdaBoostClassifier(n_estimators=200, random_state=1),
        'RF': RandomForestClassifier(n_estimators=200, random_state=1, n_jobs=-1),
        'ExtraTree': ExtraTreesClassifier(n_estimators=200, random_state=1, n_jobs=-1),
        'MLP_1': MLPClassifier(hidden_layer_sizes=(10), random_state=1),
        'MLP_2': MLPClassifier(hidden_layer_sizes=(30, 10), random_state=1),
    }



# Scoring custom
def moyenne_recall(Ytrue, Ypred):
    return (recall_score(Ytrue, Ypred) + recall_score(Ytrue, Ypred, pos_label=0)) / 2
moyenne_recall = make_scorer(moyenne_recall, greater_is_better=True)
def accuracy(Ytrue,Ypred):
    return accuracy_score(Ytrue, Ypred)
accuracy = make_scorer(accuracy, greater_is_better=True)
def moyenne_precision(Ytrue, Ypred):
    return (precision_score(Ytrue, Ypred) + precision_score(Ytrue, Ypred, pos_label=0)) / 2
moyenne_precision = make_scorer(moyenne_precision, greater_is_better=True)

# Scoring dic
scores = {
    'accuracy': accuracy_score,
    'moyenne_precision': moyenne_precision,
    'moyenne_recall': moyenne_recall
}


# Run classifier with custom parameters
def run_classifier(X, y, clfs, scoring):
    kf = KFold(n_splits=10, shuffle=True, random_state=0)
    best_model = None
    best_score = -np.inf
    best_model_name = None  # Ajouter une variable pour stocker le nom du modèle

    for name, clf in clfs.items():
        cv_acc = cross_val_score(clf, X, y, cv=kf, scoring=scoring)
        mean_score = np.mean(cv_acc)
        std_score = np.std(cv_acc)
        print(f"Score for {name} is: {mean_score:.3f} +/- {std_score:.3f}")

        if mean_score > best_score:
            best_score = mean_score
            best_model = clf
            best_model_name = name  # Mettre à jour le nom du modèle

    print(f"\nBest model: {best_model_name} with score: {best_score:.3f}")
    return best_model, best_score, best_model_name  # Retourner aussi le nom du modèle


# test for all features
def preprocess_data(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)
    return X, X_scaled, X_pca


# Save model
def save_model(model, filename="best_model.pkl"):
    with open(filename, "wb") as file:
        pickle.dump(model, file)

# Load pkl file
def load_model(filename="best_model.pkl"):
    with open(filename, "rb") as file:
        model = pickle.load(file)
    return model


def optimize_model(X, Y, best_model_name, best_model, scoring):
    # Grilles de paramètres sans préfixe `model__` pour les modèles simples
    param_grids = {
        'CART': {
            'max_depth': [None, 5, 10, 15, 20, 30],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 4]
        },
        'ID3': {
            'max_depth': [None, 5, 10, 15, 20, 30],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 4]
        },
        'Stump': {
            'max_depth': [None, 1],
            'min_samples_split': [2, 5, 10]
        },
        'KNN': {
            'n_neighbors': [3, 5, 7, 9, 11, 15],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski']
        },
        'Bag': {
            'n_estimators': [100, 200, 300, 500],
            'max_samples': [0.5, 0.75, 1.0],
            'max_features': ['sqrt', 'log2', None]
        },
        'Ad': {
            'n_estimators': [50, 100, 200, 300],
            'learning_rate': [0.001, 0.01, 0.1, 0.5, 1.0]
        },
        'RF': {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'criterion': ['gini', 'entropy']
        },
        'ExtraTree': {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'criterion': ['gini', 'entropy']
        },
        'MLP_1': {
            'hidden_layer_sizes': [(10,), (20,), (30,), (50, 20), (100, 50)],
            'alpha': [0.0001, 0.001, 0.01, 0.1],
            'learning_rate_init': [0.001, 0.01, 0.1]
        },
        'MLP_2': {
            'hidden_layer_sizes': [(10,), (20,), (30,), (50, 20), (100, 50)],
            'alpha': [0.0001, 0.001, 0.01, 0.1],
            'learning_rate_init': [0.001, 0.01, 0.1]
        }
    }
    
    # Vérifier si le modèle a une grille de paramètres
    if best_model_name in param_grids:
        param_grid = param_grids[best_model_name]
        print(f"Optimizing {best_model_name} with grid: {param_grid}")
        
        # GridSearchCV avec la grille de paramètres et scoring
        grid_search = GridSearchCV(best_model, param_grid, scoring=scoring, cv=5, n_jobs=-1)
        grid_search.fit(X, Y)
        
        print(f"Best parameters for {best_model_name}: {grid_search.best_params_}")
        
        # Retourner le meilleur modèle optimisé
        return grid_search.best_estimator_
    else:
        print(f"Pas de grille de paramètres définie pour {best_model_name}.")
        return best_model




def selection_variables(X,Y):
    # utilisation du random forest pour obtenir l'importance des features
    clf = RandomForestClassifier(n_estimators=1000,random_state=1)
    clf.fit(X, Y)
    # fonction pour afficher l'importances des features 
    importances=clf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in clf.estimators_],axis=0)
    # Sort des variables d'importances dans l'ordre décroissant 
    sorted_idx = np.argsort(importances)[::-1]
    return sorted_idx

# Nombre de variables selection
def nb_variables(X,Y,features, best_model, sorted_idx, scoring):
    # Initialisation
    scores = np.zeros(len(features))
    kf = KFold(n_splits=10, shuffle=True, random_state=0)
    # Boucle sur le nombre de variables à sélectionner
    for f in range(0, X.shape[1]):
        cv_moy_recall = cross_val_score(best_model, X[:,sorted_idx[:f+1]], Y, cv=kf, scoring=scoring, n_jobs=-1)
        scores[f] = np.mean(cv_moy_recall)
    
    return scores

# Main function
# Main function
def automate_pipeline(X, y, features, scoring):
    print("Start pipeline")

    # Step 1: Preprocess data
    print('Step 1: Preprocessing')
    X_normal, X_scaled, X_pca = preprocess_data(X)

    # Step 2: Get classifiers
    print('Step 2: Get classifiers')
    clfs = get_clfs()

    # Step 3: Test classifiers on all datasets
    print('Step 3: Test classifiers')
    datasets = {"Normal": X_normal, "Scaled": X_scaled, "PCA": X_pca}
    best_overall_model = None
    best_overall_score = -np.inf
    best_dataset_name = None
    best_model_name = None

    for dataset_name, dataset in datasets.items():
        print(f"\nTesting on {dataset_name} dataset")
        best_model, best_score, model_name = run_classifier(dataset, y, clfs, scoring)

        if best_score > best_overall_score:
            best_overall_score = best_score
            best_overall_model = best_model
            best_dataset_name = dataset_name
            best_model_name = model_name

    print(f"\nBest overall model: {best_model_name} on {best_dataset_name} dataset with score: {best_overall_score:.3f}")

    # Step 4: Optimize best model
    print('Step 4: Optimizing best model')
    optimized_model = optimize_model(X, y, best_model_name, best_overall_model, scoring)

    # Step 5: Feature selection
    print('Step 5: Feature selection')
    sorted_idx = selection_variables(X, y)
    scores = nb_variables(X, y, features, optimized_model, sorted_idx, scoring)

    # Step 6: Save model
    print('Step 6: Save optimized model')
    save_model(optimized_model)

    print("Pipeline completed successfully.")
    return optimized_model, scores




# Run pipeline
def run_all(X, y, features, scoring):
    model, score = automate_pipeline(X, y, features, scoring)
    print(f"Best model: {model}, Best score ({scoring}): {score}")
    return model, score
