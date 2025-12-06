import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import TYPE_CHECKING

# TYPE_CHECKING pour le typage statique
if TYPE_CHECKING:
    from linear_regression import LinearRegression


@dataclass
class Results:
    """
    Cette classe regroupe les résultats de la régression et calcule des métriques
    Attributs:
        model: Instance du modèle entraîné
        R2: Coefficient de détermination
        y_true: Valeurs réelles
        y_pred: Valeurs prédites
    """

    model: 'LinearRegression'
    R2: float
    y_true: np.ndarray
    y_pred: np.ndarray

    def __post_init__(self):
        """Convertit les données en array 1D"""
        self.y_true = np.asarray(self.y_true).flatten()
        self.y_pred = np.asarray(self.y_pred).flatten()
        if len(self.y_true) != len(self.y_pred):
          raise ValueError("y_true et y_pred n'ont pas la même longueur")

    def mse(self) -> float:
        """ Calcule le Mean Squared Error"""
        return np.mean((self.y_true - self.y_pred) ** 2)

    def rmse(self) -> float:
        """ Calcule le Root Mean Squared Error"""
        return np.sqrt(self.mse())

    def residuals(self) -> np.ndarray:
        """ Calcule les erreurs/ résidus """
        return self.y_true - self.y_pred

    def summary(self) -> str:
        """Donne un résumé détaillé des résultats"""
        res = self.residuals()
        mae = np.mean(np.abs(res))
        max_error = np.max(np.abs(res))

        # Interprétation du R^2
        if self.R2 >= 0.9:
            performance = "Excellent"
        elif self.R2 >= 0.7:
            performance = "Bon"
        elif self.R2 >= 0.5:
            performance = "Moyen"
        else:
            performance = "Faible"
        summary = f"""

                                Résultats de la régression linéaire

I - Performance du modèle
  • R^2 (coefficient de détermination) : {self.R2:.4f} ({performance})

II - Métriques d'erreur
  • MSE  (Mean Squared Error)       : {self.mse():.4f}
  • RMSE (Root Mean Squared Error)  : {self.rmse():.4f}
  • MAE  (Mean Absolute Error)      : {mae:.4f}
  • Erreur maximale                 : {max_error:.4f}

III - Statistiques des prédictions
  • Nombre d'échantillons            : {len(self.y_true)}
  • Moyenne des valeurs réelles      : {np.mean(self.y_true):.4f}
  • Moyenne des valeurs prédites     : {np.mean(self.y_pred):.4f}
  • Écart-type des erreurs           : {np.std(res):.4f}

IV - Coefficients du modèle
"""
        for name, coef in self.model.to_dict().items():
            summary += f"  • {name:20s} : {coef:>10.4f}\n"

        return summary

    def predictions_dataframe(self) -> pd.DataFrame:
        """Retourne un DataFrame avec les prédictions et erreurs"""
        res = self.residuals()
        return pd.DataFrame({
            'y_true': self.y_true,
            'y_pred': self.y_pred,
            'error': res,
            'abs_error': np.abs(res)})