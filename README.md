Agent IA Pong

Introduction

Agent IA Pong est un projet visant à développer une intelligence artificielle capable de jouer au jeu classique Pong en utilisant des techniques d'apprentissage par renforcement. L'agent est entraîné à l'aide d'un réseau de neurones profond (DQN) pour apprendre des stratégies efficaces en interagissant avec l'environnement de jeu.

Fonctionnalités

Environnement de Jeu Personnalisé : Simulation de Pong développée avec Pygame.

Agent Intelligent : Utilisation du Deep Q-Learning pour une prise de décision adaptée.

Entraînement Accéléré : Support de l'entraînement parallèle pour une optimisation rapide.

Mode Joueur vs Agent : Possibilité de jouer contre l'agent entraîné.

Visualisation des Performances : Suivi des métriques d'entraînement avec TensorBoard.

Installation

Prérequis

Python 3.7 ou version supérieure

pip (installateur de paquets Python)

Installation des Dépendances

Clonez le dépôt et installez les bibliothèques requises :

git clone https://github.com/crd78/agent-ia-pong.git
cd agent-ia-pong
pip install -r requirements.txt

Configuration

Assurez-vous que les répertoires pour les modèles et les logs existent :

mkdir -p models runs

Utilisation

Entraînement de l'Agent

Pour entraîner l'agent, exécutez le script train.py :

python train.py

Les modèles entraînés seront sauvegardés dans le répertoire models.

Les journaux d'entraînement seront enregistrés dans runs.

Jouer contre l'Agent

Assurez-vous d'avoir un modèle entraîné (pong_agent_best.pth) et lancez le jeu en exécutant main.py :

python main.py

Contrôles

Flèche Haut : Déplacer la raquette vers le haut

Flèche Bas : Déplacer la raquette vers le bas

Échap : Quitter le jeu

Visualisation des Performances

Utilisez TensorBoard pour suivre l'évolution de l'entraînement :

tensorboard --logdir=runs

Ensuite, ouvrez votre navigateur à l'adresse indiquée pour visualiser les graphiques des récompenses, pertes, etc.