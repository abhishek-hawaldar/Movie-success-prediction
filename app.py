#from flask import request
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#import itertools
#import tkinter as tk
from sklearn.metrics import accuracy_score
#import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import tree
from sklearn.model_selection import train_test_split  
from sklearn.naive_bayes import GaussianNB

dataset = pd.read_csv("movie.csv");
dataset.plot(kind= 'box' , figsize = (20, 10))
print ("Dataset Lenght:: ", len(dataset))

X = dataset.iloc[:, 0:-1] #contains  columns i.e attributes
y = dataset.iloc[:, -1] #contains labels
M = dataset.loc[:,'Profit']
N = dataset.loc[:,'class']

dataset.loc[dataset['Profit'] == 0, 'Profit'] = dataset['Profit'].mean()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state =30)  
resultSet=[]

def knn():
    classifier = KNeighborsClassifier(n_neighbors=6)
    classifier.fit(X_train, y_train)  #the model gets train using this 

    #make predictions
    y_pred = classifier.predict(X_test)  



def prediction():
      print("GENRE:\n 1: HORROR \n2: COMEDY \n3:ACTION \n4:ROMANCE \n5:THRILLER" ) 
      values=float(input("enter genre "))
      print (values)
      values1=float(input("enter profit(in crores)  "))
      print (values1)
      values2=float(input("enter rating  "))
      print (values2)
      predictionSet=[]
      df = pd.read_csv('movie.csv')
      df.drop('Film',axis=1,inplace=True)
      features = list(df.columns[1:4])
      target = df.columns[4:5]
      X = df[features] #our features that we will use to predict Y
      Y = df[target]
      X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=40)
      clf_knn=KNeighborsClassifier(n_neighbors=5)
      clf_knn.fit(X_train, y_train.values.ravel())
      y_pred_knn=clf_knn.predict([[values,values1,values2]])
      print (y_pred_knn)
      if y_pred_knn==1:
          res_knn="Hit"
      else:
          res_knn="Flop"
      print(res_knn)
prediction()
knn()

