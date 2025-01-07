# KID Analyzer

Un outil simple pour analyser les documents d'informations clés (KID) des produits financiers.

## Structure

```
kid_analyzer/
├── kid_analyzer.py      # Le script principal
├── requirements.txt     # Les dépendances
├── README.md           # Documentation
└── XS1914695009-EN.pdf # Exemple de fichier PDF
```

## Installation

1. Installer les dépendances :
```bash
pip install -r requirements.txt
```

2. Assurez-vous que Tesseract OCR est installé sur votre système :
```bash
sudo apt-get install tesseract-ocr
```

## Utilisation

1. Placez votre fichier KID (format PDF) dans le même dossier que le script
2. Exécutez le script :
```bash
python kid_analyzer.py
```

Le script va :
1. Convertir le PDF en images
2. Extraire le texte avec OCR
3. Analyser le contenu avec un modèle de langage
4. Sauvegarder les résultats dans `kid_analysis_raw.txt`

## Résultats

Les résultats sont sauvegardés dans un fichier texte avec une structure claire :

1. PRODUCT IDENTIFICATION
   - Nom du produit
   - Code ISIN
   - Fabricant
   - Contact

2. RISK ASSESSMENT
   - Niveau de risque
   - Description

3. PERFORMANCE SCENARIOS
   - Montant d'investissement
   - Scénarios (stress, défavorable, modéré, favorable)

4. COSTS
   - Coûts d'entrée
   - Coûts de sortie
   - Coûts annuels

5. RECOMMENDED HOLDING PERIOD
   - Période
   - Conditions de sortie anticipée
