#libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, RidgeCV, Lasso
from sklearn.preprocessing import StandardScaler
#from sklearn.datasets import load_boston
from sklearn.datasets import fetch_california_housing
   
from sklearn.datasets import fetch_openml
housing = fetch_openml(name="house_prices", as_frame=True)

print(housing.DESCR)
print(housing.frame.head())
print(housing.target.head())
print(housing.frame.info())

print("type: ", type(housing))

df = pd.DataFrame(data = housing.data, columns = housing.feature_names)
df = df.loc[:, ['LotArea', 'GrLivArea', 'TotRmsAbvGrd', 'OverallQual', 'GarageArea']]
df['Price'] = housing.target
print(df)

#Exploration
plt.figure(figsize = (10, 10))
sns.heatmap(df.corr(), annot = True)
#plt.show()

#pairplot
sns.pairplot(df)
#plt.show()

#preview
features = df.columns[0:4]
target = df.columns[-1]

#X and y values
X = df[features].values
y = df[target].values

#splot
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=17)

print("The dimension of X_train is {}".format(X_train.shape))
print("The dimension of X_test is {}".format(X_test.shape))
#Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Model
lr = LinearRegression()

#Fit model
lr.fit(X_train, y_train)

#predict
#prediction = lr.predict(X_test)

#actual
actual = y_test

train_score_lr = lr.score(X_train, y_train)
test_score_lr = lr.score(X_test, y_test)

print("The train score for lr model is {}".format(train_score_lr))
print("The test score for lr model is {}".format(test_score_lr))


#Ridge Regression Model
ridgeReg = Ridge(alpha=10)

ridgeReg.fit(X_train,y_train)

#train and test scorefor ridge regression
train_score_ridge = ridgeReg.score(X_train, y_train)
test_score_ridge = ridgeReg.score(X_test, y_test)

print("\nRidge Model............................................\n")
print("The train score for ridge model is {}".format(train_score_ridge))
print("The test score for ridge model is {}".format(test_score_ridge))
