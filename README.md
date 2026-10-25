# Georef Desktop (Windows 11)

Géoréférence une image (JPG/PNG) sur une carte, avec points de contrôle, et exporte des tuiles XYZ + un fichier MBTiles
(ouvrable dans QGIS). Les résultats sont dans `Documents\Georef\<id>\`.

## Obtenir le .exe
1. Poussez ce dossier sur GitHub.
2. Onglet **Actions** → workflow *Build Windows exe* → télécharger l'artefact **Georef-windows** (Georef.exe).

## Lancer depuis les sources (Windows)
    pip install -r requirements.txt
    python launcher.py

Internet est nécessaire pour afficher le fond de carte OpenStreetMap.
