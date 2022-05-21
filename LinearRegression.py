import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# reading dataset
data = pd.read_csv('dataset.csv')
data.head()
data.rename(columns={
    'Gender': 'gender',
    'Age Range': 'age',
    'Head Size(cm^3)': 'head_size',
    'Brain Weight(grams)': 'brain_weight'
}, inplace=True)
data.head()
data.isnull().sum()

X = data['head_size'].values
Y = data['brain_weight'].values
sns.scatterplot(X, Y)
plt.show()

x_mean = np.mean(X)
y_mean = np.mean(Y)
print("x mean "+np.float64(x_mean).astype(str))
print("y mean " +np.float64(y_mean).astype(str))

n = data.shape[0]

# Find coefficient and intercept value
numerator = 0
denominator = 0
for i in range(n):
  numerator += ( (X[i]- x_mean) * (Y[i] - y_mean) )
  denominator += (X[i]- x_mean)**2

  coeff = numerator/denominator
  intercept = y_mean - coeff * x_mean

print("coffecient or slop: "+np.float64(coeff).astype(str) )
print("Intercept: "+np.float64(intercept).astype(str))

y_pred = coeff * X + intercept

sns.scatterplot(X, Y)
sns.lineplot(X, y_pred, color='r')
plt.show()

prediction = (coeff * 4513 + intercept)
print("Predit head size 4513 to be weighted as:  " + np.float64(prediction).astype(str) )








