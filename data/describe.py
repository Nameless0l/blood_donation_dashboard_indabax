import pandas as pd
import os

# Remplacez par le chemin de votre fichier
file_path = "Updated Challenge dataset.xlsx"

# Lister toutes les feuilles
excel_file = pd.ExcelFile(file_path)
sheet_names = excel_file.sheet_names
print(f"Feuilles dans le fichier: {sheet_names}")

# Explorer chaque feuille
for sheet in sheet_names:
    df = pd.read_excel(file_path, sheet_name=sheet)
    print(f"\n--- Feuille: {sheet} ---")
    print(f"Nombre de lignes: {len(df)}")
    print(f"Colonnes: {df.columns.tolist()}")
    print("Premiers enregistrements:")
    print(df.head(2).transpose())

# Sauvegarder un rapport pour référence
with open("structure_excel.txt", "w") as f:
    f.write(f"Feuilles dans le fichier: {sheet_names}\n")
    for sheet in sheet_names:
        df = pd.read_excel(file_path, sheet_name=sheet)
        f.write(f"\n--- Feuille: {sheet} ---\n")
        f.write(f"Nombre de lignes: {len(df)}\n")
        f.write(f"Colonnes: {df.columns.tolist()}\n")
        f.write("Types de données:\n")
        for col, dtype in df.dtypes.items():
            f.write(f"  - {col}: {dtype}\n")