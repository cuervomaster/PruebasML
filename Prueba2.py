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

housing.plot(kind="scatter", x="longitude", y="latitude", grid=True, alpha=0.2)
save_fig("bad_visualization_plot")  # extra code
plt.show()