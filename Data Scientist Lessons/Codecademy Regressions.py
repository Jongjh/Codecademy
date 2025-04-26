# -*- coding: utf-8 -*-

#Linear Regression Practice 1
import codecademylib3_seaborn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv("https://content.codecademy.com/programs/data-science-path/linear_regression/honeyproduction.csv")

print(df.head())

prod_per_year = df.groupby('year').totalprod.mean().reset_index()

# Extracting the year column for X values
x = prod_per_year.year

try:
  x = x.values.reshape(-1,1)
except Exception as e:
  print(f'Reshaping error: {e}')

y = prod_per_year.totalprod

plt.scatter(x,y)
plt.show()

regr = LinearRegression()
regr.fit(x,y)

print(regr.coef_[0])
print(regr.intercept_)

y_predict = regr.predict(x)

plt.scatter(x, y_predict, color='red')
plt.show()

x_future = np.array(range(2013,2051))

try:
  x_future = x_future.reshape(-1,1)
except Exception as e:
  print(f'Reshaping error: {e}')

future_predict = regr.predict(x_future)

plt.scatter(x_future, future_predict,color='green')
plt.show()

