############### Acquisition et d'Initialisation des données #################


import pandas as pd
import urllib.request
import os

# 1. Créer le dossier data s'il n'existe pas
if not os.path.exists('../data'):
    os.makedirs('../data')

# 2. L'adresse du fichier sur internet
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00519/heart_failure_clinical_records_dataset.csv"

# 3. Destination sur ton ordinateur
destination = "../data/heart.csv"

# 4. Téléchargement magique
try:
    urllib.request.urlretrieve(url, destination)
    print("Succès ! Le fichier est maintenant dans ton dossier /data")
    
    # 5. Vérification : On l'ouvre pour voir
    df = pd.read_csv(destination)
    display(df.head())
except Exception as e:
    print(f"Erreur : {e}")
 # On recrée un DataFrame complet avec les données équilibrées
df_final = pd.DataFrame(X_res, columns=X.columns)
df_final['DEATH_EVENT'] = y_res   
# --- ÉTAPE 2 : LA FONCTION D'OPTIMISATION ---
import pandas as pd

def optimize_memory(df):
    """
    Réduit la taille des données en ajustant les types numériques.
    """
    for col in df.columns:
        col_type = df[col].dtype
        
        # Optimisation pour les nombres entiers
        if str(col_type).startswith('int'):
            df[col] = pd.to_numeric(df[col], downcast='integer')
        
        # Optimisation pour les nombres décimaux
        elif str(col_type).startswith('float'):
            df[col] = pd.to_numeric(df[col], downcast='float')
            
    return df
# --- ÉTAPE 3 : RÉSULTAT FINAL ---
# Application de la fonction
df_optimise = optimize_memory(df)

print("🚀 RÉSULTAT APRÈS OPTIMISATION")
print("-" * 50)

# Vérification des nouveaux types (int8, int16, float32, etc.)
print("Nouveaux types de données :")
print(df_optimise.dtypes)

# Calcul du gain
mem_finale = df_optimise.memory_usage(deep=True).sum() / 1024
gain = ((mem_initiale - mem_finale) / mem_initiale) * 100

print("-" * 50)
print(f"Nouvel espace mémoire : {mem_finale:.2f} KB")
print(f"Réduction totale : {gain:.1f} %")
