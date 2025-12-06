import numpy as np
import pandas as pd
from typing import List, Optional


class Dataset:
    """
    La classe permet de de gérer et valider les données pour la régression linéaire
    Attributs:
        X : Matrice des features (n × p)
        y : Vecteur des résultats (n × 1)
        features_name : Liste des noms des colonnes (longueur = p)
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, features_name: Optional[List[str]] = None):
        """
        Arguments : 
            X: Matrice des features
            y: Vecteur cible
            features_name: Noms des colonnes (auto-généré si None)
        """

        self.X = np.asarray(X, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64)

        # Vérification : validité de y avant le reshape
        if self.y.ndim > 2 or (self.y.ndim == 2 and self.y.shape[1] != 1):
            raise ValueError(f"y doit être un vecteur ou une matrice colonne (reçu {self.y.shape})")

        # Normalisation des dimensions
        if self.X.ndim == 1:
            self.X = self.X.reshape(-1, 1)
        if self.y.ndim == 1:
            self.y = self.y.reshape(-1, 1)

        # Gestion des noms des coefficients
        if features_name is None:
            self.features_name = [f"feature_{i}" for i in range(self.X.shape[1])]
        else:
            self.features_name = features_name.copy()

        self._validate()

    def _validate(self) -> None:
        """Valide la cohérence des données """

        if self.X.shape[0] != self.y.shape[0]:
            raise ValueError(
                f"X et y doivent avoir le même nombre d'échantillons. "
                f"Actuellement : X={self.X.shape[0]}, y={self.y.shape[0]}")

        if len(self.features_name) != self.X.shape[1]:
            raise ValueError(
                f"features_name a ({len(self.features_name)} éléments) "
                f"mais X a({self.X.shape[1]}) colonnes.")

        # Vérification : Dataset non vide
        if self.X.shape[0] == 0 or self.X.shape[1] == 0:
            raise ValueError("Le dataset ne peut pas être vide")

        # Vérification : Valeurs valides
        if np.isnan(self.X).any():
            raise ValueError("X contient des valeurs NaN")
        if np.isinf(self.X).any():
            raise ValueError("X contient des valeurs infinies")

        if np.isnan(self.y).any():
            raise ValueError("y contient des valeurs NaN")
        if np.isinf(self.y).any():
            raise ValueError("y contient des valeurs infinies")

    def add_intercept(self) -> 'Dataset':
        """ Ajoute une colonne d'intercept (= colonne de 1) au début de X si non présente """
        
        # Vérification : intercept existe ou non (colonne de 1)
        if self.X.shape[1] > 0 and np.allclose(self.X[:, 0], 1):
            return self

        intercept_column = np.ones((self.X.shape[0], 1))
        self.X = np.hstack([intercept_column, self.X])
        self.features_name = ["intercept"] + self.features_name

        return self

    def to_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame(self.X, columns=self.features_name)
        df['target'] = self.y
        return df


