# %% [markdown]
# # Projet 3 - Anticipez les besoins en consommation de bâtiments
# 
# ## Analyse exploratoire et création d'un dataset clean
# 
# Le but de ce notebook est d'analyser le jeu de données initial et de le traiter afin de produire un dataset "clean", exporté en csv, qui sera la base du travail de machine learning consécutif.

# %%
import pandas as pd
from MLUtils import DataAnalysis, DataEngineering
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

import warnings
warnings.filterwarnings("ignore")

# %%
df = pd.read_csv('data/2016_Building_Energy_Benchmarking_20240624.csv')

# %%
df.info()

# %% [markdown]
# Le jeu initial de données contient 3376 observations réparties en 46 colonnes/variables.

# %%
DataAnalysis.show_columns_population(df, type='bar')

# %% [markdown]
# On constate que plusieurs colonnes contiennent trop peu de données pour être exploitées. Nous enlevons donc les colonnes qui ont moins de 30% de données.

# %%
(df, logs, n_columns_removed) = DataEngineering.remove_columns_by_percentage(df, 0.3)

print("Nombre de colonnes supprimées : ", n_columns_removed)

logs

# %%
DataAnalysis.show_columns_population(df, type='matrix')

# %%
df.describe()

# %% [markdown]
# Grâce à cette analyse, nous pouvons voir que :
# - La colonne DataYear semble contenir toujours la même valeur
# - Les colonne OSEBuildingID, PropertyName, Address, City, State, TaxParcelIdentificationNumber, CouncilDistrictCode ne seront pas utile pour nos algorithmes, car bien trop spécifiques
# 
# Nous les enlevons donc du dataset.

# %%
df = DataEngineering.remove_columns_by_name(df, ['OSEBuildingID', 'DataYear', 'PropertyName', 'Address', 'City', 'State', 'TaxParcelIdentificationNumber', 'CouncilDistrictCode'])

# %% [markdown]
# ## Analyse des colonnes de type number et valeurs aberrantes

# %%
# On liste les colonnes qui ont des valeurs de type number
numericColumns = df.select_dtypes(include=['number']).columns

# %%
import matplotlib.pyplot as plt
import math

data_to_plot = [df[col].dropna() for col in numericColumns]

# Calcule du nombre de lignes nécessaires
num_rows = math.ceil(len(numericColumns) / 2)

fig, axs = plt.subplots(num_rows, 2, figsize=(12*2, 4*num_rows))
axs = axs.ravel()

for idx, col in enumerate(numericColumns):
    axs[idx].boxplot(data_to_plot[idx], vert=True, patch_artist=True)
    axs[idx].set_title(f'Diagramme à moustache de {col}')
    axs[idx].set_ylabel('Valeurs')
    axs[idx].set_xticks([])

# Supprimer les axes non utilisés s'il y en a
for idx in range(len(numericColumns), num_rows*2):
    axs[idx].axis('off')

plt.tight_layout()
plt.show()


# %%
# Créer un dataframe ne contenant que les colonnes de type number
df_num = df.select_dtypes(include=['number'])

# %%
correlation_matrix = df_num.corr()

# On sauvegarde la matrice de corrélation
correlation_matrix.to_csv('data/correlation_matrix.csv')

print(correlation_matrix)

# %% [markdown]
# ## Visualisation de la matrice de correlation

# %%
sns.heatmap(correlation_matrix, annot=True)
plt.show()

# %% [markdown]
# ## Analyse des colonnes contenant des valeurs autres que des numbers

# %%
# create a dataframe with columns which are not number type
df_not_num = df.select_dtypes(exclude=['number'])

# %%
df_not_num.info()

# %%
df_not_num.sample(5)

# %%
plt.figure(figsize=(10, 8))
count_plot = sns.countplot(x='BuildingType', data=df_not_num)
count_plot.set_xticklabels(count_plot.get_xticklabels(), rotation=45, horizontalalignment='right')
plt.tight_layout()

# %% [markdown]
# ## Nous allons simplifier la colonne 'BuildingType' en classant les valeurs en 2 catégories :
# - 'Multifamily' : valeur 0
# - 'Autres' : valeur 1

# %%
# Nous allons simplifier la colonne 'BuildingType' en classant les valeurs en 2 catégories
df['BuildingType'].unique()

# %%
multifamily_values = ['Multifamily LR (1-4)', 'Multifamily MR (5-9)', 'Multifamily HR (10+)']
non_multifamily_values = ['NonResidential', 'Nonresidential COS', 'Nonresidential WA', 'SPS-District K-12', 'Campus']

# replace values in column 'BuildingType'
df['BuildingType'] = df['BuildingType'].replace(multifamily_values, 'Multifamily')

# test wether the value in column 'BuildingType' is 0. If not, replace by 1
df['BuildingType'] = df['BuildingType'].apply(lambda x: 'Other' if x != 'Multifamily' else x)


# %%
# Vérifions que les valeurs sont bien uniquement des 0 et des 1
df['BuildingType'].unique()

# %%
encoder = OneHotEncoder()
encoded_data = encoder.fit_transform(df[['BuildingType']])
encoded_df = pd.DataFrame(encoded_data.toarray(), columns=[f"BuildingType_{category}" for category in encoder.categories_[0]])
df = pd.concat([df, encoded_df], axis=1)

# %% [markdown]
# Les valeurs de BuildingType sont maintenant remplacées.

# %%
plt.figure(figsize=(10, 8))
count_plot = sns.countplot(x='PrimaryPropertyType', data=df_not_num)
count_plot.set_xticklabels(count_plot.get_xticklabels(), rotation=45, horizontalalignment='right')
plt.tight_layout()

# %%
# list possible values of column 'PrimaryPropertyType'
df['PrimaryPropertyType'].unique()

# %%
# Lists for each category
residential_buildings = ["Low-Rise Multifamily", "Mid-Rise Multifamily", "High-Rise Multifamily", "Senior Care Community", "Residence Hall"]
commercial_office_buildings = ["Hotel", "Small- and Mid-Sized Office", "Large Office", "Retail Store", "Medical Office", "Restaurant", "Laboratory"]
educational_healthcare_facilities = ["K-12 School", "University", "Hospital"]
industrial_special_purpose = ["Warehouse", "Distribution Center", "Refrigerated Warehouse", "Self-Storage Facility", "Mixed Use Property", "Supermarket / Grocery Store", "Worship Facility", "Office", "Other"]

# Function to map property type to a category number
def property_type_to_number(property_type):
    if property_type in residential_buildings:
        return 'Residential'
    elif property_type in commercial_office_buildings:
        return 'Commercial'
    elif property_type in educational_healthcare_facilities:
        return 'EducationalHealthcare'
    elif property_type in industrial_special_purpose:
        return 'IndustrialOther'
    else:
        return 'IndustrialOther'  # For any property type that doesn't fit into these categories

df['PrimaryPropertyType'] = df['PrimaryPropertyType'].apply(lambda x: property_type_to_number(x))


# %%
# list possible values of column 'PrimaryPropertyType'
df['PrimaryPropertyType'].unique()

# %%
encoder = OneHotEncoder()
encoded_data = encoder.fit_transform(df[['PrimaryPropertyType']])
encoded_df = pd.DataFrame(encoded_data.toarray(), columns=[f"PrimaryPropertyType_{category}" for category in encoder.categories_[0]])
df = pd.concat([df, encoded_df], axis=1)

# %%
plt.figure(figsize=(10, 8))
count_plot = sns.countplot(x='Neighborhood', data=df_not_num)
count_plot.set_xticklabels(count_plot.get_xticklabels(), rotation=45, horizontalalignment='right')
plt.tight_layout()

# %%
# list possible values of column 'Neighborhood'
df['Neighborhood'].unique()

# %%
# Function to normalize and regroup neighborhood names
def normalize_neighborhood(neighborhood):
    # Normalize the case (convert all to uppercase)
    normalized_neighborhood = neighborhood.upper()

    # Special handling for 'DELRIDGE' to include 'DELRIDGE NEIGHBORHOODS'
    if 'DELRIDGE' in normalized_neighborhood:
        return 'DELRIDGE'

    return normalized_neighborhood

# Assuming you have a DataFrame 'df' with a column 'Neighborhood'
# Apply the function to the DataFrame
df['Neighborhood'] = df['Neighborhood'].apply(lambda x: normalize_neighborhood(x))


# %%
# list possible values of column 'Neighborhood'
df['Neighborhood'].unique()

# %%
encoder = OneHotEncoder()
encoded_data = encoder.fit_transform(df[['Neighborhood']])
encoded_df = pd.DataFrame(encoded_data.toarray(), columns=[f"Neighborhood_{category}" for category in encoder.categories_[0]])
df = pd.concat([df, encoded_df], axis=1)

# %%
plt.figure(figsize=(10, 8))
count_plot = sns.countplot(x='LargestPropertyUseType', data=df_not_num)
count_plot.set_xticklabels(count_plot.get_xticklabels(), rotation=45, horizontalalignment='right')
plt.tight_layout()

# %%
plt.figure(figsize=(10, 8))
count_plot = sns.countplot(x='SecondLargestPropertyUseType', data=df_not_num)
count_plot.set_xticklabels(count_plot.get_xticklabels(), rotation=45, horizontalalignment='right')
plt.tight_layout()

# %%
plt.figure(figsize=(10, 8))
count_plot = sns.countplot(x='DefaultData', data=df_not_num)
count_plot.set_xticklabels(count_plot.get_xticklabels(), rotation=45, horizontalalignment='right')
plt.tight_layout()

# %%
plt.figure(figsize=(10, 8))
count_plot = sns.countplot(x='ComplianceStatus', data=df_not_num)
count_plot.set_xticklabels(count_plot.get_xticklabels(), rotation=45, horizontalalignment='right')
plt.tight_layout()

# %% [markdown]
# ## Grâce aux analyses des champs non numériques, nous pouvons exclure les champs suivants :
# - ComplianceStatus
# - DefaultData
# - LargestPropertyType, SecondLargestPropertyType et SecondLargestPropertyUseTypeGFA : ces colonnes sont inutiles étant donné que nous avons pris en compte PrimaryPropertyType.
# - ListOfAllPropertyUseTypes est inutile dans le contexte
# - ZipCode, Latitude et Longitude : il a été décidé de garder le quartier (Neighborhood) et donc les coordonnées exactes des batiments devient obsolète.
# 
# Ces champs sont trop peu diversifié pour pouvoir être utiles dans nos modèles.
# Les autres champs présentent pour certains de nombreuses valeurs possibles. Nous allons plus tard essayer de les combiner afin d'avoir un champ unique plus exploitable.

# %%
# array of columns to remove
columns_to_remove = [
    'ComplianceStatus',
    'DefaultData',
    'LargestPropertyUseType',
    'SecondLargestPropertyUseType',
    'SecondLargestPropertyUseTypeGFA',
    'Neighborhood',
    'ListOfAllPropertyUseTypes',
    'Latitude',
    'Longitude',
    'ZipCode',
    'BuildingType',
    'PrimaryPropertyType'
]

df = DataEngineering.remove_columns_by_name(df, columns_to_remove)

# %%
# Analyse après nettoyage et engineering des données
DataAnalysis.show_columns_population(df, type='matrix')

# %%
# create a variable containing columns names without ENERGYSTARScore
columns_without_energystarscore = df.columns.drop('ENERGYSTARScore')
# eliminate rows with missing values for columns_without_energystarscore
df = df.dropna(subset=columns_without_energystarscore)


# %%
DataAnalysis.show_columns_population(df, type='matrix')

# %%
for col in df.columns:
    print(f"{col}: {type(df[col].iloc[0])}")

# %%
df.info()

# %%
correlation_matrix_clean = df.corr()

# write correlation matrix to file
correlation_matrix_clean.to_csv('data/correlation_matrix_clean.csv')

# %%
# Set the size of the plot
# Note: The size is set in inches and the dpi (dots per inch) determines the resolution.
# For a 1920x1080 resolution with a typical screen dpi of 96, use the following dimensions:
plt.figure(figsize=(1920/96, 1400/96), dpi=96)

# Create the heatmap
sns.heatmap(correlation_matrix_clean, annot=True)

# Show the plot
plt.show()


# %%
# Transformer la colonne NumberofFloors en logarithme afin de réduire l'effet des outliers
# Définir une fonction pour appliquer le logarithme en toute sécurité
def safe_log(x, min_val=0.0001):
    return np.log(x + min_val)

# Appliquer la fonction logarithmique sécurisée
df['NumberofFloors'] = df['NumberofFloors'].apply(safe_log)


# %%
# Extraire les corrélations avec 'SiteEnergyUse(kBtu)'
# correlations = correlation_matrix_clean['SiteEnergyUse(kBtu)']

# # Définir un seuil de corrélation, par exemple 0.75
# threshold = 0.75

# # Identifier les variables fortement corrélées (à l'exclusion de la variable elle-même)
# strongly_correlated = correlations[abs(correlations) > threshold].drop(['SiteEnergyUse(kBtu)', 'TotalGHGEmissions'])
# #strongly_correlated.drop('TotalGHGEmissions')

# # Afficher les variables fortement corrélées
# print(strongly_correlated)

# # Supprimer ces variables du dataset
# df = df.drop(columns=strongly_correlated.index)


# %% [markdown]
# ### Création d'une nouvelle colonne "Age" qui est une transformation de la colonne YearBuilt (2023 - YearBuilt)

# %%
df['Age'] = 2017 - df['YearBuilt']

df.drop(columns='YearBuilt', inplace=True)

# %% [markdown]
# ### Utilisation du logarithme pour la colonne "Age" qui est une transformation qui permet de mieux exploiter cette valeur par la suite
# 

# %%
# Make Age a logarithmic feature
df['Age'] = df['Age'].apply(safe_log)

# %% [markdown]
# #### Construction d'une nouvelle variable, qui sera ratio d'utilisation d'énergie par âge

# %%
# remove rows where 'SiteEUI(kBtu/sf)' is 0
df = df[df['SiteEUI(kBtu/sf)'] != 0]

df['EnergyUse_Age_Ratio'] = df['SiteEUI(kBtu/sf)'] / df['Age']

# %% [markdown]
# ## On réaffiche les correlations pour voir si les nouvelles colonnes ont un impact

# %%
plt.figure(figsize=(1920/96, 1400/96), dpi=96)

# Create the heatmap
sns.heatmap(correlation_matrix_clean, annot=True)

# Show the plot
plt.show()

# %% [markdown]
# ## Génération du fichier csv clean pour les modèles de machine learning.

# %%
df.sample(5)

# %%
# remove rows which contain NaN values
df = df.dropna()

# %%
df.info()

# %%
# write the resulting dataframe to a csv file
df.to_csv('data/clean.csv', index=False)


