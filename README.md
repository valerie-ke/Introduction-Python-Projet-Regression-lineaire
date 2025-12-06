# Projet Python - Régression linéaire 

## Description
## Contexte & objectif

Ce projet implémente un modèle de régression linéaire from scratch en utilisant la méthode des moindres carrés ordinaires (OLS).

**Objectif**: Créer un système de régression linéaire complet (sans utiliser sklearn pour les calculs), permettant de prédire une variable cible à partir de features numériques.

Il est structuré autour de trois classes principales et d'une fonction pipeline qui orchestre l'ensemble.

---
## Structure du projet

| Module | Description | Responsabilités |
|--------|-------------|-----------------|
| **Dataset** | Gestion des données | • Validation des données (NaN, Inf, dimensions)<br>• Ajout automatique de l'intercept<br>• Conversion en DataFrame Pandas |
| **LinearRegression** | Modèle de régression | • Calcul des coefficients : $\beta = (X^T X)^{-1} X^T y$<br>• Prédictions sur nouvelles données<br>• Export des coefficients en dictionnaire |
| **Results** | Analyse des résultats | • Calcul des métriques ($R^2$, MSE, RMSE, MAE)<br>• Génération de rapports détaillés<br>• DataFrame des prédictions et erreurs |
| **pipeline.py** | Orchestration | • Fonction `linear_regression_pipeline()`<br>• 4 exemples/démonstrations <br>• Point d'entrée principal |
| **test.py** | Validation | • 10 tests unitaires automatisés<br>• Comparaison avec sklearn<br>• Tests des cas limites |

---
## Les différents tests 
Le projet contient différents tests couvrant différents aspects :

| Test| Description| Validation|
| - | - | - |
| Test 1| Pipeline complet | Intégration globale |
| Test 2a | Ajout intercept| Fonctionnalité Dataset  |
| Test 2b | Idempotence intercept| Appel répété ne change pas les données|
| Test 3  | Une seule feature  | Cas limite |
| Test 4  | Grand dataset | Dataset 1000 échantillons |
| Test 5  | Conversion coefficients | `to_dict()` retourne le dictionnaire correct |
| Test 6  | Prédiction sur nouvelles données | Inférence vecteur/matrice |
| Test 7  | Gestion d'erreurs | Exceptions levées correctement (3 cas) |
| Test 8  | Validation sklearn | Comparaison exactitude coefficients et $R^2$ |
| Test 9  | Prédiction list / 1D  | Fonctionne avec list et vecteur 1D  |


---

## Robustesse et validation

Le modèle est bien conçu pour être fiable et robuste :
* **Validations automatiques** : Vérification des dimensions, détection des NaN/Inf, cohérence des données
* **Validations des calculs** : Comparaison avec sklearn, tests sur différentes tailles de datasets, et vérification de la formule de régression
* **Gestion d'erreurs** : Messages clairs en cas de problème (multicolinéarité, dimensions incorrectes, modèle non entraîné)
* **Tests avec assertions** : 9 tests automatisés pour garantir le bon fonctionnement de chaque module
* **Analyse complète** : Le rapport est complet et inclut les métriques d'erreur (MSE, RMSE, MAE), les prédictions ...
---

## Auteur
Valérie Kessavane 

Master 1 Économétrie et Statistiques  
