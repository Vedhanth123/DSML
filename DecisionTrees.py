# importing dataset from sklearn
from sklearn.datasets import load_iris
# importing decision tree algorithm
from sklearn.tree import DecisionTreeClassifier
# importing RandomForst algorithm
from sklearn.ensemble import RandomForestClassifier
# LabelEncoder is used to map names to values
from sklearn.preprocessing import LabelEncoder
# This module helps to split our dataset into two parts 1 for testing and 2 for training
from sklearn.model_selection import train_test_split
# This tells us how accurate our model is
from sklearn.metrics import accuracy_score

import pandas as pd

iris = load_iris()

x = iris.data
y = iris.target

print(x)
print('-----------------------------------------------------------------------------------------------------------------------------------------------')
print(y)

# 0 represents iris setosa, 1 represents iris vercicolour, 2 represents iris ----
flowers = {0: 'Iris-Setosa', 1: 'Iris-Versicolor', 2: 'Iris:Veriginica'}

dtc = DecisionTreeClassifier()

dtc.fit(x, y)

# gives the answer to the question by predicting
dtc.predict([[2.3, 2.1, 0.7, 0.2]])

# gives the dimensions of the array
x.shape

# creating a dataframe of inputs
df = pd.DataFrame(x)

print(df)
print('----------------------------------------------------------------------------------------------------------------------------------------')

# adding heading
df.columns = ['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width']

df['Flower_Class'] = y


print(df)
print("-------------------------------------------------------------------------------------------------------------------------------------")


# adding another coloum which says which FlowerClass name is over there as 0,1,2
df['Flower'] = df['Flower_Class'].apply(lambda x: flowers[x])

# removing a column from the dataframe [It doesn't actually remove the column cause pandas is extra carefull and create a temporary variable and stores it and deletes it from the main table. To avoid this add a parameter called inplace=True]
df.drop(['Flower_Class'], axis=1, inplace=True)


print(df.shape)

y = df['Flower']

x = df[['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width']].values

# splitting data for testing and testing
# we are splitting 10% of the data from the dataset
# we use random_state parameter to use random data from the dataset and that dataset shouldn't chage all the time
xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.1, random_state=0)

dtc.fit(xtr, ytr)

dtc_pred = dtc.predict(xte)

accuracy = accuracy_score(yte, dtc_pred)

print(dtc_pred)
print(yte)
print(f"The accuracy of the Decision Tree Classifier is {round(accuracy * 100, 2)}%")
print('------------------------------------------------------------------------------------------------------------------------------')

rfc = RandomForestClassifier()

rfc.fit(xtr, ytr)

rfc_pred = rfc.predict(xte)

rfc_acc = accuracy_score(yte, rfc_pred)

print(f"The accuracy score for Random Forest model is {round(rfc_acc*100, 2)}%")
