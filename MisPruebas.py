from pathlib import Path
import pandas as pd
import tarfile
import urllib.request
import matplotlib.pyplot as plt
import numpy as np

# Crear una Serie a partir de una lista
datos = [1, 3, 5, 7, 9]
serie = pd.Series(datos)
print(serie)

# # Crear una Serie con índices personalizados
# indices = ['a', 'b', 'c', 'd', 'e']
# serie_con_indices = pd.Series(datos, index=indices)
# print(serie_con_indices)

# # Crear una Serie a partir de un diccionario
# diccionario = {'a': 1, 'b': 2, 'c': 3}
# serie_desde_diccionario = pd.Series(diccionario)
# print(serie_desde_diccionario)

# Operaciones en Series
serie_duplicada = serie * 2
print(serie_duplicada)

# Acceso a elementos
elemento = serie[2]  # Accede al tercer elemento
print(elemento)

# Slicing
subserie = serie[1:4]  # Accede a una porción de la Serie
print(subserie)
