import math
import requests
from bs4 import BeautifulSoup
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
num_records = 200
#123
url = \
'http://wiki.stat.ucla.edu/socr/index.php/SOCR_Data_Dinov_020108_HeightsWeights'
page = requests.get(url)
height_weight_df = pd.read_html(url)[1][['Height(Inches)','Weight(Pounds)']]

x = height_weight_df['Height(Inches)'].values.reshape(num_records, 1)
y = height_weight_df['Weight(Pounds)'].values.reshape(num_records, 1)

model = linear_model.LinearRegression().fit(x,y)

print("ŷ =" + str(model.intercept_[0]) + " + " + str(model.coef_.T[0][0]) + " x₁")

y_pred = model.predict(x)
print(mean_absolute_error(y, y_pred))

mae = mean_absolute_error(y, y_pred)

plt.scatter(x, y, color='black')
plt.plot(x, y_pred, color='blue', linewidth=3)
plt.plot(x, y_pred + mae, color='lightgray')
plt.plot(x, y_pred - mae, color='lightgray')
plt.show()

# corr, pval = pearsonr(x[:,0], y[:,0])
# print(corr)
# print(pval < 0.05)




