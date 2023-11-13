# Linear Regression on Boston Housing Dataset 
Overview:
This project is a basic example of a machine learning linear regression task using the Boston Housing dataset. The goal is to predict the median value of owner-occupied homes (MEDV) based on various features.

## Dependencies:

* NumPy
  
* pandas
  
* scikit-learn

## Usage:

* Make sure you have the required dependencies installed

```bash

pip install -r requirements.txt

```
* Copy the Python code provided and run it to perform the linear regression analysis

```python

import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import load_boston
df = load_boston()
df.keys()
boston = pd.DataFrame(df.data, columns=df.feature_names)
boston.head()
boston['MEDV'] = df.target
boston.head()
boston.isnull()
boston.isnull().sum()
from sklearn.model_selection import train_test_split
X = boston.drop('MEDV', axis=1)
Y = boston['MEDV']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.15, random_state = 5 )

print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
lin_model = LinearRegression()

lin_model.fit(X_train, Y_train)
y_train_predict = lin_model.predict(X_train)
rmse = (np.sqrt(mean_squared_error(Y_train, y_train_predict)))

print("the model performance for training set")
print('RMSE is {}'.format(rmse))
print("\n")

y_test_predict = lin_model.predict(X_test)
rmse = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))

print("the model performance for testing set")
print('RMSE is {}'.format(rmse))
```

## Model Evaluation 

The model's performance is evaluated using the root mean squared error (RMSE) on both the training and testing sets.

```bash
the model performance for training set
RMSE is 4.710901797319796

the model performance for testing set
RMSE is 4.687543527902972
```
