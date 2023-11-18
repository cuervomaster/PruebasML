import pandas as pd

# Crear un DataFrame de ejemplo
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]}, index=['x', 'y', 'z'])

# Utilizando loc
subset_loc = df.loc['x':'y']

# Utilizando iloc
subset_iloc = df.iloc[0:2]

print("Subset usando loc:")
print(subset_loc)

print("\nSubset usando iloc:")
print(subset_iloc)
