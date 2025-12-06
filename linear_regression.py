import numpy as np
from typing import List, Dict, Union, Optional

class LinearRegression:
    """
    La classe implémente un modèle de régression linéaire par la méthode des moindres carrés
    Attributs:
        coefficients: Vecteur des coefficients du modèle
        features_names: Noms des features associés aux coefficients
    """

    def __init__(self):
        self.coefficients: Optional[np.ndarray] = None
        self.features_names: Optional[List[str]] = None
        self._is_fitted: bool = False

    def fit(self, X: np.ndarray, y: np.ndarray, features_names: Optional[List[str]] = None) -> 'LinearRegression':
        """
        Entraîne le modèle en calculant les coefficients
        Arguments:
            X: Matrice des features (n × p)
            y: Vecteur cible
            features_names: Noms des features (optionnel)
        """

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1, 1)

        # Gestion des noms de features
        if features_names is None:
            self.features_names = [f"feature_{i}" for i in range(X.shape[1])]
        else:
            if len(features_names) != X.shape[1]:
                raise ValueError(
                    f"features_names doit correspondre au nombre de colonnes dans X " f"({X.shape[1]} attendu, {len(features_names)} reçu).")
            self.features_names = features_names.copy()

        # Résolution de l'équation normale B = (X^T X)^(-1) X^T y
        X_transpose = X.T
        XtX = X_transpose @ X
        Xty = X_transpose @ y

        # Vérification : XtX inversible
        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            raise ValueError("La matrice X^T X n'est pas inversible (colonnes colinéaires ?)")

        self.coefficients = (XtX_inv @ Xty).flatten()

        self._is_fitted = True
        return self


    def predict(self, X: Union[np.ndarray, list]) -> np.ndarray:
        """Prédit les valeurs pour de nouveaux échantillons X en les multipliant par les coefficients du modèle"""

        if not self._is_fitted:
            raise RuntimeError("Le modèle doit être entraîné avant de faire des prédictions => Appelez d'abord fit()" )

        X = np.asarray(X, dtype=np.float64)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        if X.shape[1] != len(self.coefficients):
            raise ValueError(
                f"Dimensions incompatibles : le modèle attend {len(self.coefficients)} colonnes,"f"reçu {X.shape[1]}")

        return X @ self.coefficients

    def to_dict(self) -> Dict[str, float]:
        """Retourne les coefficients sous forme de dictionnaire"""
        if not self._is_fitted:
            raise RuntimeError("Le modèle doit être entraîné avant d'accéder aux coefficients !")

        return dict(zip(self.features_names, self.coefficients))
    


  