import numpy as np
import pandas as pd
from sklearn.datasets import make_regression 
from sklearn.linear_model import LinearRegression as SklearnLR 
from sklearn.metrics import r2_score 
from typing import List, Dict, Union, Tuple, Optional

from dataset import Dataset
from linear_regression import LinearRegression
from results import Results

def linear_regression_pipeline(dataset: 'Dataset') -> 'Results':
    """Fonction qui va permettre d'exécuter le pipeline complet : 
    ajoute l'intercept, entraîne le modèle, calcule les prédictions et retourne les résultats """

    dataset.add_intercept()

    model = LinearRegression()
    model.fit(dataset.X, dataset.y, dataset.features_name)

    y_pred = model.predict(dataset.X)
    y_true = dataset.y.flatten()

    # R^2 = 1 - (SS_res / SS_tot)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    # Gestion du cas où ss_tot = 0 (= toutes les valeurs de y identiques)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

    return Results(model=model,R2=r2,y_true=y_true,y_pred=y_pred)

# Utilisation
def demonstration():
    """ Démonstration du pipeline avec affichage des résultats """
    
    print("="*70)

    # Exemple 1 : Régression linéaire multiple
    print("\n Exemple 1 : Régression linéaire multiple (3 features)")
    
    X, y = make_regression(n_samples=100, n_features=3, n_informative=3, noise=10, random_state=42)
    dataset = Dataset(X=X, y=y, features_name=["F1", "F2", "F3"])
    
    print(f"Dataset : {dataset}")
    print(f"\nAperçu des données :")
    print(dataset.to_dataframe().head())
    
    results = linear_regression_pipeline(dataset)
    
    print(results.summary())
    
    print("\nExemple de prédictions :")
    print(results.predictions_dataframe().head(10).to_string())
    
    # Exemple 2 : Régression linéaire simple (1 feature)
    print("\n" + "="*70)
    print("Exemple 2 : Régression linéaire simple (1 feature)")
    
    X_one = np.random.rand(50, 1)
    y_one = 3 * X_one[:, 0] + 2 + np.random.randn(50) * 0.5
    dataset_one = Dataset(X_one, y_one, features_name=["variable_x"])
    
    print(f"Dataset : {dataset_one}")
    print(f"Relation théorique : y = 3x + 2 + bruit")
    
    results_one = linear_regression_pipeline(dataset_one)
    
    print(results_one.summary())
    
    coeffs = results_one.model.to_dict()
    print(f"\nÉquation trouvée : y = {coeffs['intercept']:.2f} + {coeffs['variable_x']:.2f}*x")
    print(f"(Attendu : y = 2.00 + 3.00*x)")
    
    # Exemple 3 : Prédiction sur nouvelles données
    print("\n" + "="*70)
    print("Exemple 3 : Prédiction sur nouvelles données")    
    np.random.seed(123)
    n_features = len(results.model.coefficients) - 1
    new_X = np.hstack([[1], np.random.randn(n_features)]).reshape(1, -1)
    new_pred = results.model.predict(new_X)
    
    print(f"Nouvelle observation : {new_X[0, 1:]}")
    print(f"Prédiction : {new_pred[0]:.4f}")

    # Prédictions multiples
    n_samples = 5
    new_X_multi = np.hstack([
        np.ones((n_samples, 1)), 
        np.random.randn(n_samples, n_features)
    ])
    pred_multi = results.model.predict(new_X_multi)
    
    print(f"\nPrédictions multiples ({n_samples} échantillons) :")
    for i, pred in enumerate(pred_multi):
        print(f" Échantillon {i+1} : {pred:.4f}")

    # Exemple 4 : Validation avec sklearn
    print("\n" + "="*70)
    print("Exemple 4 : Validation avec sklearn")
    
    X_val, y_val = make_regression(n_samples=200, n_features=3, random_state=42)
    dataset_val = Dataset(X_val, y_val)
    results_val = linear_regression_pipeline(dataset_val)
    
    # Comparaison avec sklearn
    X_with_intercept = dataset_val.X
    sklearn_model = SklearnLR(fit_intercept=False)
    sklearn_model.fit(X_with_intercept, y_val)
    r2_sklearn = r2_score(y_val, sklearn_model.predict(X_with_intercept))
    
    print(f"R^2 modèle implémenté : {results_val.R2:.10f}")
    print(f"R^2 sklearn          : {r2_sklearn:.10f}")
    print(f"Différence          : {abs(results_val.R2 - r2_sklearn):.2e}")
    
    print("\n" + "="*70)
    print("Fin")
    print("="*70)

if __name__ == "__main__":
    demonstration()
