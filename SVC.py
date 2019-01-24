# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 11:12:30 2019

@author: Diip
"""

from sklearn.model_selection import train_test_split 
from sklearn.svm import SVC 
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



# 02. load data  
df = pd.read_csv(filepath_or_buffer ="pima-indians-diabetes.txt") 

df

X = df.drop(['diabetes'],axis=1)
y = df['diabetes']

X_train,X_test, y_train,y_test = train_test_split(X,y,test_size = 0.3, random_state=5)

X_test.shape

svc_model = SVC()
svc_model.fit(X_train,y_train)
y_pred = svc_model.predict(X_test)

cm = confusion_matrix(y_test,y_pred)

sns.heatmap(cm, annot = True)




min_train = X_train.mean()
range_train = (X_train-min_train).max()

X_train_scaled = (X_train-min_train)/range_train

sns.set()

sns.scatterplot(x = X_train['pregnancies'],y = X_train['diastolic'],hue = y_train, data=df)

















