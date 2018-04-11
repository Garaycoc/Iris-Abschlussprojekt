# IrisAbschlussprojekt
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 12:49:04 2018

@author: maxim
"""

import pandas as pd
from sklearn import model_selection
from sklearn import neighbors
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

path = "C:/Users/maxim/OneDrive/BWL/Semester 5/Python/Abschlussprojekt/"

df_original = pd.read_csv(path + "iris.csv")

df = shuffle(df_original)

print(df)

Species = df.Species

predictors = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']

#exogene Variable wird bestimmt
X = df[predictors]

#endogene Variable wird bestimmt
Y = Species

#Trainings- und Testdaten werden festgelegt
Xtrain, Xtest, ytrain, ytest = model_selection.train_test_split(X, Y, test_size=0.2)

#K Nearest Neighbors Modell wird gebaut
model = neighbors.KNeighborsClassifier(n_neighbors=5)
model.fit(Xtrain, ytrain)

#1. Modell wird getestet
def Bewertung():
    test_accuracy = model.score(Xtest, ytest)
    train_accuracy = model.score(Xtrain, ytrain)
    score = model_selection.cross_val_score(model, X, Y, cv = 10)
    return print(test_accuracy, train_accuracy, score)

Bewertung()

Z = Xtest

print(model.predict(Z))

#SVM Modell wird gebaut
model2 = Pipeline([('minmaxscaler', MinMaxScaler(copy=True, feature_range=(0, 1))),('svm', SVC(C=1.0))])
model2.fit(Xtrain, ytrain)

#2. Modell wird getestet
def Bewertung2():
    test_accuracy2 = model2.score(Xtest, ytest)
    train_accuracy2 = model2.score(Xtrain, ytrain)
    score2 = model_selection.cross_val_score(model2, X, Y, cv = 10)
    return print(test_accuracy2, train_accuracy2, score2)

Bewertung2()

print(model2.predict(Z))
