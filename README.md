# Agent ia Pong

## Introduction

Agent ia Pong est un projet visant à développer une intelligence artificielle capable de jouer au jeu classique Pong en utilisant des techniques d'apprentissage par renforcement. L'agent est entraîné à l'aide d'un réseau de neurones profond (DQN) pour apprendre des stratégies efficaces en interagissant avec l'environnement de jeu.

## Fonctionnalités

- **Environnement de Jeu Personnalisé** : Développé avec Pygame pour une simulation réaliste de Pong.
- **Agent Intelligent** : Utilise Deep Q-Learning pour prendre des décisions basées sur l'état actuel du jeu.
- **Entraînement Parallèle** : Supporte l'entraînement simultané de plusieurs environnements pour accélérer le processus d'apprentissage.
- **Interface Joueur vs Agent** : Permet aux utilisateurs de jouer contre l'agent entraîné.
- **Visualisation avec TensorBoard** : Suivi des métriques d'entraînement en temps réel.

## Installation

### Prérequis

- Python 3.7 ou supérieur
- pip

### Installation des Dépendances

Clonez le dépôt et installez les dépendances requises :


git clone https://github.com/votre-utilisateur/agent-ia-pong.git
cd agent-ia-pong
pip install -r requirements.txt

Configuration
Assurez-vous que les répertoires pour les modèles et les logs existent :

mkdir models
mkdir runs

Configuration
Assurez-vous que les répertoires pour les modèles et les logs existent :

mkdir models
mkdir runs

Utilisation
Entraînement de l'Agent
Pour entraîner l'agent, exécutez le script train.py :

python train.py
Les modèles entraînés seront sauvegardés dans le répertoire models, et les journaux d'entraînement seront disponibles dans runs.

Jouer contre l'Agent
Assurez-vous d'avoir un modèle entraîné dans pong_agent_best.pth. Lancez le jeu en exécutant main.py :

Contrôles :

Flèche Haut : Déplacer la raquette vers le haut
Flèche Bas : Déplacer la raquette vers le bas
Échap : Quitter le jeu
Visualisation des Performances
Utilisez TensorBoard pour visualiser les métriques d'entraînement :

tensorboard --logdir=runs

Ouvrez votre navigateur et allez à l'adresse indiquée pour voir les graphiques des récompenses, des pertes, etc.