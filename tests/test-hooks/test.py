# -*- coding: utf-8 -*-
import json
import os
import pathlib

# Chemin du fichier JSON
chemin_fichier = pathlib.Path("donnees.json")

# Création de données
donnees = {"nom": "Alice", "age": 30, "ville": "Paris"}

# Écriture des données dans le fichier JSON
with open(chemin_fichier, "w") as fichier:
    json.dump(donnees, fichier)

# Lecture des données depuis le fichier JSON
with open(chemin_fichier, "r") as fichier:
    donnees_lues = json.load(fichier)

# Affichage des données lues
print("Données lues depuis le fichier JSON :")
print(donnees_lues)

# Exemple d'utilisation du module os
chemin_absolu = os.path.abspath(chemin_fichier)
print("Chemin absolu du fichier :", chemin_absolu)

# Exemple d'utilisation du module pathlib
fichier_pathlib = pathlib.Path(chemin_fichier)
if fichier_pathlib.exists():
    print("Le fichier existe.")
else:
    print("Le fichier n'existe pas.")
