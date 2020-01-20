#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 14:58:04 2020

@author: sanskruti
"""


# Problem Statement: Maximize Profit for Venture Capitalist Investment based on Startup Dataset
# Indpendent Variables: R&D Spend ($), Administration ($), Marketing Spend ($), State (Category)
# Dependent Variable: Profit ($)
# Dataset Size: 50, Train Test Split: 40:10
# Feature Selected for final model: R&D Spend And Marketing Spend($)


# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing dataset
dataset = pd.read_csv('50_Startups_Multiple_Linear_Regression_Sanskruti.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data: State
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

## Feature Scaling
#"""from sklearn.preprocessing import StandardScaler
#sc_X = StandardScaler()
#X_train = sc_X.fit_transform(X_train)
#X_test = sc_X.transform(X_test)
#sc_y = StandardScaler()
#y_train = sc_y.fit_transform(y_train)"""
#

## Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

## Predicting Test set results
y_pred = regressor.predict(X_test)

# Backward Elimination based Feature Selection: Statistical Significance: 0.05
# Library is used to get statistical information of independent variables
import statsmodels.formula.api as sm
# Add the column for b0 coefficient which is not supported by statsmodel
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)

X_opt = X[:, [0, 1, 2, 3, 4, 5]]
regressor_ols = sm.OLS(endog = y, exog = X_opt).fit()
print(regressor_ols.summary())

X_opt = X[:, [0, 1, 3, 4, 5]]
regressor_ols = sm.OLS(endog = y, exog = X_opt).fit()
print(regressor_ols.summary())

X_opt = X[:, [0, 3, 4, 5]]
regressor_ols = sm.OLS(endog = y, exog = X_opt).fit()
print(regressor_ols.summary())

X_opt = X[:, [0, 3, 5]]
regressor_ols = sm.OLS(endog = y, exog = X_opt).fit()
print(regressor_ols.summary())

X_opt = X[:, [0, 3]]
regressor_ols = sm.OLS(endog = y, exog = X_opt).fit()
print(regressor_ols.summary())


#
## Visualising the Training set results
#plt.scatter(X_train, y_train, color = 'red')
#plt.plot(X_train, regressor.predict(X_train), color = 'blue')
#plt.title('Salary vs Experience (Training set)')
#plt.xlabel('Years of Experience')
#plt.ylabel('Salary')
#plt.show()
#
## Visualising the Test set results
#plt.scatter(X_test, y_test, color = 'red')
#plt.plot(X_train, regressor.predict(X_train), color = 'blue')
#plt.title('Salary vs Experience (Test set)')
#plt.xlabel('Years of Experience')
#plt.ylabel('Salary')
#plt.show()