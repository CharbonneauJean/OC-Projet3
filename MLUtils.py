import pandas as pd
import missingno as msno

class DataEngineering:
    @staticmethod
    def remove_columns_by_percentage(df, percent):
        """
        Supprime les colonnes du DataFrame qui ont moins de 'percent' de valeurs renseignées.

        Parameters:
        df (DataFrame): DataFrame d'origine
        percent (float): pourcentage de remplissage minimal pour conserver une colonne (0 <= percent <= 1)

        Returns:
        DataFrame: Nouveau DataFrame avec les colonnes appropriées supprimées
        """
        if percent < 0 or percent > 1:
            raise ValueError("Le pourcentage doit être compris entre 0 et 1.")

        logs = []

        total_rows = len(df)

        columns_to_remove = []
        columns_removed_n = 0

        for column in df.columns:
            non_na_count = df[column].count()
            non_na_percent = non_na_count / total_rows
            if non_na_percent < percent:
                logs.append(f"La colonne {column} a été supprimée car elle ne contient que {round(non_na_percent * 100,2)}% de valeurs renseignées.")
                columns_to_remove.append(column)
                columns_removed_n += 1

        return (df.drop(columns=columns_to_remove), logs, columns_removed_n)
    
    @staticmethod
    def remove_columns_by_name(df, columns_to_remove):
        """
        Supprime les colonnes du DataFrame dont les noms sont dans la liste 'columns_to_remove'.

        Parameters:
        df (DataFrame): DataFrame d'origine
        columns_to_remove (list): liste des noms de colonnes à supprimer

        Returns:
        DataFrame: Nouveau DataFrame avec les colonnes appropriées supprimées
        """
        return df.drop(columns=columns_to_remove)
    
class DataAnalysis:
    @staticmethod
    def show_columns_population(df, type):
        """
        Affiche le nombre de valeurs manquantes par colonne du DataFrame.

        Parameters:
        df (DataFrame): DataFrame d'origine
        type (string): type de graphique à afficher ('matrix' ou 'bar')
        """

        # if type not matrix or bar, raise error
        if(type!='matrix' and type!='bar'):
            raise ValueError("Le type doit être soit 'matrix' soit 'bar'.")

        if(type=='matrix'):
            msno.matrix(df)
        elif(type=='bar'):
            msno.bar(df)
