import os
import re
import pandas as pd
from typing import List
import numpy as np
import pandas as pd
import numpy as np


def list_constant_columns(data: pd.DataFrame, threshold: float = 1.0, exclude_categories: bool = True, exclude_booleans: bool = True) -> List:

    excluded_types = []
    if exclude_categories:
        excluded_types.append('category')
    if exclude_booleans:
        excluded_types.append('boolean')

    selected_columns = data.select_dtypes(exclude=excluded_types).columns

    # Liste pour stocker les colonnes à supprimer
    const_cols = []

    for col in selected_columns:
        # Calculer la proportion de la valeur la plus fréquente dans la colonne
        value_frequencies = data[col].value_counts(normalize=True)

        # most_frequent_value = value_frequencies.idxmax()
        most_frequent_value_ratio = value_frequencies.max()

        # Vérifier si cette proportion dépasse le seuil (threshold)
        if most_frequent_value_ratio >= threshold:
            const_cols.append(col)
            # print(
            #     f"Column '{col}' is constant at {most_frequent_value} for {most_frequent_value_ratio:.2%} of the rows.")

    return const_cols


def clean_dataframe(data: pd.DataFrame, drop_constant_thr=1.0, exclude_categories=True, exclude_booleans=True):
    """
    Clean the DataFrame by:
    1. Dropping columns with zero variance (constant columns).
    2. Replacing NaN values with the column mean.
    3. Replacing +inf with the column max value and -inf with the column min value.
    """
    # Step 1: Drop columns with zero variance
    constant_columns = list_constant_columns(
        data=data, threshold=drop_constant_thr, exclude_categories=exclude_categories, exclude_booleans=exclude_booleans)
    df_clean = data.drop(columns=constant_columns)
    if constant_columns != []:
        print(f"Dropped columns with zero variance: {constant_columns}")

    # Step 2: Replace NaN values with column mean
    df_clean = df_clean.apply(lambda x: x.fillna(x.mean()) if x.dtype in [
                              np.float64, np.float32, np.int64] else x)

    # Step 3: Replace +inf with max and -inf with min
    df_clean = df_clean.apply(lambda x: x.replace([np.inf], x[np.isfinite(x)].max(
    )) if x.dtype in [np.float64, np.float32, np.int64] and np.isinf(x).any() else x)
    df_clean = df_clean.apply(lambda x: x.replace([-np.inf], x[np.isfinite(x)].min(
    )) if x.dtype in [np.float64, np.float32, np.int64] and np.isinf(x).any() else x)

    return df_clean


def supprimer_fichier_feather_recursif(dossiers):
    """
    Parcourt les dossiers et leurs sous-dossiers récursivement et supprime
    tous les fichiers 'data.feather' s'ils existent.

    :param dossiers: Liste de chemins de dossiers à parcourir.
    """

    # Définir l'expression régulière pour correspondre à 'data.feather' ou 'data_<word>.feather'
    pattern = re.compile(
        r'^data(?:_[a-zA-ZçÇéÉèÈêÊëËàÀâÂîÎïÏôÔûÛùÙüÜœŒ\s\W]+)*\.feather$', re.UNICODE)
    print("Deleting files...")
    for dossier in dossiers:
        # Parcours récursivement le dossier et ses sous-dossiers
        for chemin_racine, sous_dossiers, fichiers in os.walk(dossier):
            # Filtre les fichiers correspondant à l'expression régulière
            for fichier in fichiers:
                # print(fichier)
                if pattern.match(fichier):

                    chemin_fichier = os.path.join(chemin_racine, fichier)
                    try:
                        # Supprime le fichier
                        os.remove(chemin_fichier)
                        print(f"Fichier supprimé: {chemin_fichier}")
                    except Exception as e:
                        print(
                            f"Erreur lors de la suppression de {chemin_fichier}: {e}")
