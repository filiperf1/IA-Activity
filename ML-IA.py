# Imports
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification


#Loading dataset
filename = 'processed_data.csv'
data = pd.read_csv(filename)


#Assing Input and Output values
X = data.data
Y = data.target


#Building the classification model
clf = RandomForestClassifier()
clf.fit(X,Y)

#Split the data set 70/30%

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

#Rebuilding the model
clf.fit(X_train, Y_train)

#Model performance output
print(clf.score(X_test, Y_test))