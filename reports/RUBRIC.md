# Grille de notation PlantGuard

| Catégorie | Critère | Preuve à présenter |
| --- | --- | --- |
| Code Quality | Versioning Git | Historique des commits, différence avec `main`, workflows associés à une révision |
| Code Quality | Structure modulaire | `frontend/src`, `backend/app`, `ml/model.py`, `ml/preprocessing`, `ml/training`, `ml/evaluation` |
| Code Quality | Tests unitaires | `uv run --locked python -m pytest`. Tests des architectures, de l’API, des fichiers rejetés, des splits, du registre et de la dérive |
| Code Quality | Documentation | `README.md` avec la procédure de reproduction, les commandes de démonstration et les sources |
| Reproductibilité | Environnements virtuels | `.python-version` et `uv sync --locked` créent un environnement isolé |
| Reproductibilité | Gestion des dépendances | `pyproject.toml`, `uv.lock` et `frontend/package-lock.json`. CI et images installent depuis les verrous |
| Reproductibilité | Seeds et randomness | `params.yaml`, répartition déterministe, seeds Python/NumPy/PyTorch et générateur du DataLoader |
| Reproductibilité | Configuration externalisée | `params.yaml` pour DVC et les expériences. Variables d’environnement et `.env.example` pour le service |
| Data Management | Versioning des données | `data/raw/plantvillage.dvc`, `dvc.lock`, fetcher épinglé et `dvc status` |
| Data Management | Validation des données | `data/processed/validation.json` avec effectifs, images rejetées et doublons. Tests des chemins et formats |
| Data Management | Pipeline de preprocessing | `dvc dag` et `dvc repro`. Déduplication, séparation par feuille connue, augmentation d’entraînement et transformation d’inférence commune |
| Data Management | Sécurisation des données | Données et secrets exclus de Git, remote DVC local et contrôlé, limites d’envoi/décodage, modèle monté en lecture seule, télémétrie agrégée |
| Model Management | Versioning des modèles | `models/registry/`, empreinte SHA-256, seuils de promotion, pointeur de service et procédure de retour à une version antérieure |
| Model Management | Tracking des expériences | MLflow, paramètres, split, métriques par époque, rapports et artefacts des checkpoints |
| Model Management | Métriques et validation | `reports/metrics.json`, `reports/evaluation.json`, validation pour la sélection et test figé pour l’évaluation |
| Deployment | Containerisation | Dockerfiles d’API et de frontend, Compose, healthcheck, proxy et utilisateur sans privilèges |
| Deployment | CI/CD Pipeline | `.github/workflows/ci.yml` et `cd.yml`. Tests, couverture, smoke test Docker, publication GHCR par SHA |
| Deployment | Monitoring en production | Prometheus, compteurs d’inférence, histogramme de latence, version, dérive de luminosité et règles d’alerte dans `monitoring/` |

## Présentation

1. Exécuter les tests et afficher le DAG DVC.
2. Comparer le CNN et MobileNet dans MLflow. Montrer le split commun et la version promue.
3. Afficher les métriques du test figé et la matrice de confusion.
4. Démarrer Compose. Téléverser une image d’une classe du benchmark, puis afficher les compteurs et le modèle actif dans Prometheus.
5. Ouvrir les workflows GitHub Actions et montrer le rapport de vérification. Une exécution de livraison demande un tag de version ou un déclenchement manuel.

La grille totalise 54 points. Le barème et la note finale relèvent de l’évaluateur.
