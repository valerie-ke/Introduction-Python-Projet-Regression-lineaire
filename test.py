import numpy as np
import pandas as pd
from sklearn.datasets import make_regression 
from sklearn.linear_model import LinearRegression as SklearnLR 
from sklearn.metrics import r2_score 

from dataset import Dataset
from linear_regression import LinearRegression
from results import Results
from pipeline import linear_regression_pipeline

# Tests via assert

# Test 1 : principal
def test_pipeline_complet():
    """ Test d'intégration du pipeline complet """
    X, y = make_regression(n_samples=100, n_features=3, n_informative=3, noise=10, random_state=42)
    dataset = Dataset(X=X, y=y, features_name=["F1", "F2", "F3"])
    results = linear_regression_pipeline(dataset)
    
    assert results.model._is_fitted, "Le modèle devrait être entraîné"
    assert len(results.model.coefficients) == 4, f"Attendu 4 coefficients, reçu {len(results.model.coefficients)}"
    assert 0 <= results.R2 <= 1, f"R^2 hors limites : {results.R2}"
    assert len(results.y_true) == len(results.y_pred), "y_true et y_pred doivent avoir la même longueur"

# Test 2a
def test_add_intercept():
    """ Test de l'ajout automatique de l'intercept """
    X_simple = np.random.rand(5, 2)
    dataset = Dataset(X_simple, np.random.rand(5))
    
    shape_before = dataset.X.shape
    dataset.add_intercept()
    shape_after = dataset.X.shape
    
    assert shape_after[1] == shape_before[1] + 1, "Une colonne devrait avoir été ajoutée"
    assert dataset.features_name[0] == "intercept", "Le premier nom devrait être 'intercept'"
    assert np.allclose(dataset.X[:, 0], 1), "La première colonne devrait contenir des 1"

# Test 2b
def test_add_intercept_idempotence():
    """Test de l'idempotence de add_intercept (pas de doublon)."""
    X = np.random.rand(5, 2)
    dataset = Dataset(X, np.random.rand(5))
    dataset.add_intercept()
    
    shape_before = dataset.X.shape
    dataset.add_intercept()  
    dataset.add_intercept()  # Appel répété
    shape_after = dataset.X.shape
    
    assert shape_after == shape_before, "Appels répétés ne doivent pas ajouter de colonnes"
    assert dataset.features_name.count("intercept") == 1, "Un seul intercept doit être présent"

# Test 3 
def test_regression_une_feature():
    """Test avec une seule feature """
    X_one = np.random.rand(50, 1)
    y_one = 3 * X_one[:, 0] + 2 + np.random.randn(50) * 0.5
    dataset = Dataset(X_one, y_one)
    results = linear_regression_pipeline(dataset)
    
    assert len(results.model.coefficients) == 2, "Devrait avoir 2 coefficients (intercept + 1 feature)"
    assert results.R2 > 0.5, f"R^2 devrait être > 0.5, reçu {results.R2:.4f}"

# Test 4 
def test_dataset_large():
    """ Test avec un grand dataset """
    X_large, y_large = make_regression(n_samples=1000, n_features=5,  noise=20, random_state=1)
    # petit bruit -> noise = 20 (à faire varier mais donc changer l'assert car R^2 = 0,8 plutôt strict)
    dataset = Dataset(X_large, y_large)
    results = linear_regression_pipeline(dataset)
    
    assert len(results.model.coefficients) == 6, "Devrait avoir 6 coefficients (5 features + intercept)"
    assert results.R2 > 0.8, f"R^2 devrait être > 0.8, reçu {results.R2:.4f}"

#  Test 5 
def test_coefficients_to_dict():
    """Test de la conversion coefficients -> dictionnaire."""
    X, y = make_regression(n_samples=50, n_features=2, random_state=42)
    dataset = Dataset(X, y)
    results = linear_regression_pipeline(dataset)
    
    coef_dict = results.model.to_dict()
    
    assert isinstance(coef_dict, dict), "Doit retourner un dictionnaire"
    assert all(name in coef_dict for name in results.model.features_names), "Tous les noms de features doivent être présents"
    assert all(isinstance(v, (float, np.floating)) for v in coef_dict.values()), "Toutes les valeurs doivent être des floats"
    
#  Test 6 
def test_prediction_nouvelles_donnees():
    """ Test de prédiction sur nouvelles données """
    X, y = make_regression(n_samples=100, n_features=3, random_state=42)
    dataset = Dataset(X, y)
    results = linear_regression_pipeline(dataset)
    
    np.random.seed(123)
    n_features = len(results.model.coefficients) - 1
    
    # 1 seule prédiction
    new_X = np.hstack([[1], np.random.randn(n_features)]).reshape(1, -1)
    new_pred = results.model.predict(new_X)
    
    assert new_pred.shape == (1,), "Devrait retourner un vecteur de longueur 1"
    assert not np.isnan(new_pred[0]), "La prédiction ne devrait pas être NaN"
    
    # Prédictions multiples
    n_samples = 3
    new_X_multi = np.hstack([np.ones((n_samples, 1)), np.random.randn(n_samples, n_features)])
    pred_multi = results.model.predict(new_X_multi)
    
    assert pred_multi.shape == (n_samples,), f"Devrait retourner {n_samples} prédictions"
    assert not np.isnan(pred_multi).any(), "Aucune prédiction ne devrait être NaN"

#  Test 7 
def test_exceptions():
    """ Test de la gestion des exceptions """
    # Test 7a : predict avant fit
    try:
        lr = LinearRegression()
        lr.predict(np.array([[1, 2, 3]]))
        assert False, "Devrait lever une RuntimeError"
    except RuntimeError:
        pass  # Comportement attendu
    
    # Test 7b : features_names mauvaise taille
    try:
        lr = LinearRegression()
        lr.fit(np.random.randn(50, 3), np.random.randn(50), 
               features_names=["a", "b"])
        assert False, "Devrait lever une ValueError"
    except ValueError:
        pass  
    
    # Test 7c : X avec mauvaise dimension pour predict
    try:
        lr = LinearRegression()
        lr.fit(np.random.randn(50, 3), np.random.randn(50))
        lr.predict(np.array([[1, 2]]))
        assert False, "Devrait lever une ValueError"
    except ValueError:
        pass

# Test 8 
def test_validation_sklearn():
    """ Test de validation avec sklean """
    X_val, y_val = make_regression(n_samples=1000, n_features=3, random_state=42)
    
    # Modèle codé
    dataset = Dataset(X_val, y_val)
    results = linear_regression_pipeline(dataset)
    
    # Avec Sklearn
    X_with_intercept = dataset.X
    sklearn_model = SklearnLR(fit_intercept=False)
    sklearn_model.fit(X_with_intercept, y_val)
    y_pred_sklearn = sklearn_model.predict(X_with_intercept)
    r2_sklearn = r2_score(y_val, y_pred_sklearn)
    
    assert np.allclose(results.R2, r2_sklearn, atol=1e-10), f"R^2 différent de sklearn : {abs(results.R2 - r2_sklearn)}"
    assert np.allclose(results.model.coefficients, sklearn_model.coef_, atol=1e-10), "Coefficients différents de sklearn"

# Test 9
def test_prediction_formats():
    """ Test de prédiction avec différents formats d'entrée """
    X = np.array([[1], [2], [3]])
    y = np.array([2, 4, 6])
    dataset = Dataset(X, y)
    dataset.add_intercept()
    
    model = LinearRegression()
    model.fit(dataset.X, dataset.y, dataset.features_name)
    
    # Test avec vecteur 1D
    X_1D = np.array([4, 5, 6])
    X_1D = np.vstack([np.ones(X_1D.shape), X_1D]).T
    y_pred_1D = model.predict(X_1D)
    
    assert y_pred_1D.shape == (3,), \
        "Prédiction 1D doit retourner vecteur longueur 3"
    
    # Test avec list
    X_list = [[1, 7], [1, 8]]
    y_pred_list = model.predict(X_list)
    
    assert y_pred_list.shape == (2,), \
        "Prédiction avec list doit retourner vecteur longueur 2"
    assert not np.isnan(y_pred_list).any(), \
        "Aucune prédiction ne doit être NaN"
    

# Exécution des tests

if __name__ == "__main__":
    test_pipeline_complet()
    test_add_intercept()
    test_add_intercept_idempotence()
    test_regression_une_feature()
    test_dataset_large()
    test_coefficients_to_dict()
    test_prediction_nouvelles_donnees()
    test_exceptions()
    test_validation_sklearn()
    test_prediction_formats()
    
    print("Tests passés avec succès")
