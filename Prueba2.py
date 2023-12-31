from pathlib import Path
import pandas as pd
import tarfile
import urllib.request
import matplotlib.pyplot as plt
import numpy as np

#DESCARGAR LA DATA
def load_housing_data():
    tarball_path = Path("datasets/housing.tgz")
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path)
        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path="datasets")
    return pd.read_csv(Path("datasets/housing/housing.csv"))

housing = load_housing_data()


#EXPLORAR LA DATA
# extra code – code to save the figures as high-res PNGs for the book
IMAGES_PATH = Path() / "images" / "end_to_end_project"
IMAGES_PATH.mkdir(parents=True, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = IMAGES_PATH / f"{fig_id}.{fig_extension}"
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

# # extra code – the next 5 lines define the default font sizes
# plt.rc('font', size=14)
# plt.rc('axes', labelsize=14, titlesize=14)
# plt.rc('legend', fontsize=14)
# plt.rc('xtick', labelsize=10)
# plt.rc('ytick', labelsize=10)

# housing.hist(bins=50, figsize=(12,8))
# save_fig("attribute_histogram_plots")  # extra code
# plt.show()

#SEPARAR DATA PARA EL TRAIN Y PARA EL TEST
def shuffle_and_split_data(data, test_ratio):
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

train_set, test_set = shuffle_and_split_data(housing, 0.2)
# print(len(train_set))
# print(len(test_set))
# print(test_set.head())
# print(train_set.head())


from zlib import crc32

def is_id_in_test_set(identifier, test_ratio):
    # crc32 e una función que cuando se le pasa el número como bytes lo convierte pseudo aleatoriamente
    # en un número de 0 a 2^32, con una distribución más o menos distribuida para un rango de números
    # por lo tanto cuando se multiplica test_ratio * 2^32 se toma una proporción definida del rango de números
    # pero seleccionados pseudo aleatoriamente
    return crc32(np.int64(identifier)) < test_ratio * 2**32

def split_data_with_id_hash(data, test_ratio, id_column):
    ids = data[id_column]
    #print(ids)
    #genera un dataframe donde reemplaza cada elemento de data[id_column] por True o False, según el resultado de aplicar
    # la función is_id_in_test_set
    in_test_set = ids.apply(lambda id_: is_id_in_test_set(id_, test_ratio))
    #print(in_test_set)
    # retorna dos dataframe, uno con los registros para el training y otro para el testing
    return data.loc[~in_test_set], data.loc[in_test_set]

# genera la columna index donde se coloca los índices de cada fila
housing_with_id = housing.reset_index()  # adds an `index` column
# print(housing_with_id.head())
#genera los dos conjuntos de datos train y test con un porcentaje de 20% y la columna index
train_set, test_set = split_data_with_id_hash(housing_with_id, 0.2, "index")

#crea una columna id en el df housing_with_id a partir de un cálculo con las columnas longitude y latitude
# del df housing, se toman los valores entre filas correspondientes
housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
#print(housing_with_id.head())

train_set, test_set = split_data_with_id_hash(housing_with_id, 0.2, "id")


from sklearn.model_selection import train_test_split
#************************************************************************
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

# print(len(train_set))
# print(len(test_set))
# print(test_set.head())
# print(train_set.head())
# print(test_set['total_bedrooms'].isnull().sum())
# print(housing.head())

#Crea una nueva columna que categoriza la columna median_income segun 5 rangos, np.inf representa un número muy grande
housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])

# # print(housing.head())
# housing["income_cat"].value_counts().sort_index().plot.bar(rot=0, grid=True)
# # print(housing.head())
# plt.xlabel("Income category")
# plt.ylabel("Number of districts")
# # save_fig("housing_income_cat_bar_plot")  # extra code
# # plt.show()

from sklearn.model_selection import StratifiedShuffleSplit
#**********************************************************

splitter = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
strat_splits = []
for train_index, test_index in splitter.split(housing, housing["income_cat"]):
    strat_train_set_n = housing.iloc[train_index]
    strat_test_set_n = housing.iloc[test_index]
    strat_splits.append([strat_train_set_n, strat_test_set_n])

strat_train_set, strat_test_set = strat_splits[0]

# print(strat_train_set.info())
# print()
# print(strat_test_set.info())

# # Cuenta la frecuencia de cada categoría en la columna 'income_cat'
# category_counts = strat_test_set['income_cat'].value_counts()

# # Ordena las categorías si es necesario
# category_counts.sort_index(inplace=True)

# # Imprime los resultados
# print(category_counts)

# category_counts = strat_train_set['income_cat'].value_counts()

# # Ordena las categorías si es necesario
# category_counts.sort_index(inplace=True)

# # Imprime los resultados
# print(category_counts)

# # primer_registro_longitude = strat_train_set['longitude'].iloc[0]
# # primer_registro_latitude = strat_train_set['latitude'].iloc[0]

# # Imprime los valores
# # print("Longitude:", primer_registro_longitude)
# # print("Latitude:", primer_registro_latitude)

#otra forma de hacer la divisió para el train y el test, una sola opción de división
#***********************************************************************************

strat_train_set, strat_test_set = train_test_split(
    housing, test_size=0.2, stratify=housing["income_cat"], random_state=42)

#Como más adelante ya no se va a utilizar la columna ´income_cat´ se borra

for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

#por si acaso para guardar el set de entrenamiento original, creamos una copia para trabajar
housing = strat_train_set.copy()

# housing.plot(kind="scatter", x="longitude", y="latitude", grid=True,
# s=housing["population"] / 100, label="population",
# c="median_house_value", cmap="jet", colorbar=True,
# legend=True, sharex=False, figsize=(10, 7))
#save_fig("bad_visualization_plot")  # extra code
#plt.show()

corr_matrix = housing.corr(numeric_only=True)
#print(corr_matrix["median_house_value"].sort_values(ascending=False))

# # Utilizando una libreria de pandas podemos hacer directamente las correlaciones cruzadas entre los campos numéricos
# # Es posible especificar los campos que nos interesa correlacionar para no generar más gráficos de los necesarios
# from pandas.plotting import scatter_matrix
# attributes = ["median_house_value", "median_income", "total_rooms",
# "housing_median_age"]
# scatter_matrix(housing[attributes], figsize=(12, 8))
# plt.show()

# #Teniendo una percepción más precisa de las posibles correlaciones y dependencias, se puede hacer el ploteo
# # de aquellas parejas que tienen una mayor correlación
# housing.plot(kind="scatter", x="median_income", y="median_house_value",
# alpha=0.1, grid=True)
# plt.show()

#Se generan columnas o campos combinados a partir de otros campos, para tener información más pertinente para el análisis
# por ejemplo el número total de habitaciones en todo el distrito no es útil, pero si el número de habitaciones por hogar
housing["rooms_per_house"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_ratio"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["people_per_house"] = housing["population"] / housing["households"]

# #Ahora volvemos a hacer el análisis de correlación con los campos combinados
# corr_matrix = housing.corr(numeric_only=True)
# print(corr_matrix["median_house_value"].sort_values(ascending=False))

# Preparando la data para el training, el dataset de entrenamiento
# quitamos la columna objetivo del dataset de entrenamiento
housing = strat_train_set.drop("median_house_value", axis=1)

# separamos la columna objetivo para luego evaluar el nivel de precisión del modelo
housing_labels = strat_train_set["median_house_value"].copy()

# #Otro punto a solucionar es verificar si hay valores ausentes, por ejemplo en la columna total_bedrooms
# # para eso hay tres opciones: 1 quitar esos distritos, 2 eliminar la columna y 3 hacer una imputación, es decir, reemplazar
# # los valores ausentes con la media, mediaan, etc.

# housing.dropna(subset=["total_bedrooms"], inplace=True) # option 1
# housing.drop("total_bedrooms", axis=1) # option 2
# median = housing["total_bedrooms"].median() # option 3
# housing["total_bedrooms"].fillna(median, inplace=True)

# En este caso vamos a hacer la opción 3 pero usando una librería sklearn.impute import SimpleImputer
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")

# antes debemos preparar el dataset solo con las columnas que tienen valores numéricos
housing_num = housing.select_dtypes(include=[np.number])

# Ahora sí podemos aplicar el imputador
imputer.fit(housing_num)
# print(imputer.statistics_)
# print(housing_num.median().values)

# Generar el dataset final de entrenamiento
# Ojo que la salida de imputer.transform(housing_num) es un arreglo de NumPy: X no tiene ni nombres de columnas ni índice
X = imputer.transform(housing_num)

# #Para darle nuevamente las propiedades de un Dataframe
# housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index)
# print(housing_tr.info()) # para ver que ahora todos sus campos tienen sus valores completos

# # MANEJO DE LOS CAMPOS CATEGÓRICOS
# #********************************

# housing_cat = housing[["ocean_proximity"]] #tomamos la columna con los valores categóricos

# # from sklearn.preprocessing import OrdinalEncoder # importamos la función
# # ordinal_encoder = OrdinalEncoder()
# # housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat) #Obtenemos el dataframe transformado donde cada valor textual es número
# # print(housing_cat_encoded[:8]) # para ver el resultado
# # print(ordinal_encoder.categories_) # para ver el orden en que ha identificado las categorías

# #Otra forma o alternativa para codificar la variable categórica con ceros y unos
# from sklearn.preprocessing import OneHotEncoder # importamos la función
# cat_encoder = OneHotEncoder()
# housing_cat_1hot = cat_encoder.fit_transform(housing_cat) # identificará cada categoría y le asignará un "bit" dentro una trama
# # print(housing_cat_1hot.toarray()) # para ver los unos y ceros en cada trama, como foquitos, representando cada categoría
# # print(cat_encoder.categories_) # para ver el orden en que ha identificado las categorías

# #Otra forma de codificar usando pandas con get_dummies()
# df_test = pd.DataFrame({"ocean_proximity": ["INLAND", "NEAR BAY"]})
# #print(pd.get_dummies(df_test).astype(int)) # genera la matriz con las dos categorias identificada, astype para que lo llene con 0 1 y no True False

# # OneHotEncoder si es capaz de manejar un valor que no aplica a ninguna categoría de las aprendidas, asociando puros ceros
# # En cambio get_dummies va a generar una nueva columna para esa categoría
# df_test_unknown = pd.DataFrame({"ocean_proximity": ["<2H OCEAN","ISLAND"]})
# cat_encoder.handle_unknown = "ignore"
# print(cat_encoder.transform(df_test_unknown).toarray())

# #conocer los nombres de la columnas o características que ha aprendido el encoder
# print(cat_encoder.feature_names_in_)

# #conocer los nombres de la columnas o características que generará al aplicar la trasnformación
# print(list(cat_encoder.get_feature_names_out()))

# #Las escalas de los atributos no son similares, 
# # por ejemplo en un caso va de 2 a 39 320
# # y en otro caso va de 0.5 a 15
# print(housing_num['total_rooms'].min())
# print(housing_num['total_rooms'].max())
# print(housing_num['median_income'].min())
# print(housing_num['median_income'].max())

# #Es una buena práctica la normalización/estandarización de la escala para poder procesar mejor
# from sklearn.preprocessing import MinMaxScaler
# min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
# housing_num_min_max_scaled = min_max_scaler.fit_transform(housing_num)
# print(housing_num_min_max_scaled[:8])
# print()

# # Utilizando el transformador StandardScaler de sklearn aplicar la estandarización
from sklearn.preprocessing import StandardScaler
# std_scaler = StandardScaler()
# housing_num_std_scaled = std_scaler.fit_transform(housing_num)
# print(housing_num_std_scaled[:8])


# Para mejor manejo de las características multimodales, a parte de la bucketización
# se puede recurrir al uso de RBF radial basis function, que es tener una función que dependen de la distancia
# entre la entrada del valor y un punto fijo de la característica
from sklearn.metrics.pairwise import rbf_kernel

#aquí se genera los valores de una función RBF usando una librer+ia de sklearn
#se hace sobre la característica de housing_median_age, para un punto fijo de 35 años
# mientras mayor es gamma, el valor decae más rápido a medida que se aleja del punto fijo
age_simil_35 = rbf_kernel(housing[["housing_median_age"]], [[35]], gamma=0.1)

# # extra code – this code generates Figure 2–18
# #*********************************************
# ages = np.linspace(housing["housing_median_age"].min(),
#                    housing["housing_median_age"].max(),
#                    500).reshape(-1, 1)
# gamma1 = 0.1
# gamma2 = 0.03
# rbf1 = rbf_kernel(ages, [[35]], gamma=gamma1)
# rbf2 = rbf_kernel(ages, [[35]], gamma=gamma2)

# fig, ax1 = plt.subplots()

# ax1.set_xlabel("Housing median age")
# ax1.set_ylabel("Number of districts")
# ax1.hist(housing["housing_median_age"], bins=50)

# ax2 = ax1.twinx()  # create a twin axis that shares the same x-axis
# color = "blue"
# ax2.plot(ages, rbf1, color=color, label="gamma = 0.10")
# ax2.plot(ages, rbf2, color=color, label="gamma = 0.03", linestyle="--")
# ax2.tick_params(axis='y', labelcolor=color)
# ax2.set_ylabel("Age similarity", color=color)

# plt.legend(loc="upper left")
# save_fig("age_similarity_plot")
# plt.show()
# #*************************************************************************


# #Escalar los valores objetivos puede ser una opción en aquellos casos en que presentan una distribución 
# # por ejemplo de cola pesada
# # Se cuenta con la función inverse_transform del escalador, que permite, luego de obtener la predicción
# # en la escala ajustada, volver a los valores reales y poder evaluar el nivel de precisión de la predicción
# # respecto de los valores esperados
# from sklearn.linear_model import LinearRegression
# target_scaler = StandardScaler()
# scaled_labels = target_scaler.fit_transform(housing_labels.to_frame())
# model = LinearRegression()
# model.fit(housing[["median_income"]], scaled_labels)
# some_new_data = housing[["median_income"]].iloc[:5] # pretend this is new data
# print(housing[["median_income"]].iloc[:5])
# scaled_predictions = model.predict(some_new_data)
# predictions = target_scaler.inverse_transform(scaled_predictions)
