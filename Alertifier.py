#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  5 18:21:01 2019

@author: has
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import pickle
import warnings
from sklearn.model_selection import train_test_split
from matplotlib import pyplot
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from pyod.utils.utility import standardizer
from sklearn.metrics import accuracy_score
warnings.filterwarnings("ignore")

class Alertifier:
    
    def __init__(self,filename,agent_ref):
        self.filename=filename
        self.agent_ref=agent_ref
        self.model=None
        self.X_train=None
        self.X_test=None
        self.y_train=None
        self.y_test=None 
        self.y_pred=None

   
        
    def filter(self):     
        bd = pd.read_pickle(self.filename)
    #data=load("/home/has/Airline/dm-pfe-hm/d","rb")
    #bd
        
        df=pd.DataFrame(bd)
        df=df.loc[df['details_agent'] == self.agent_ref]
        df['B_dept']=(df.details_flights_departure-df.details_validation_at)
        #df.dropna()
        df['B_dept']=df['B_dept']/np.timedelta64(1,'h')
        #df=df[df.details_status =='TKTT']
        df['d']=df.details_validation_at.dt.date
        df['t']=df.details_validation_at.dt.time
        df=df[df.details_status =='TKTT']
        df['B_dept']=round(df.B_dept,0)
        df=df.drop(df[df.B_dept < 0 ].index)
        df=df.drop(df[df.details_price <400].index)    
        X=df.iloc[:,[2,5]].values
        #from sklearn.preprocessing import MinMaxScaler
        #scaler = MinMaxScaler()
        #X=scaler.fit_transform(X)
        
        X = standardizer(X)
       # self.x=X
        return X
    
    def GetLabel(self,X):
    
        '''------------------OSVM--------------------------'''
        
        from sklearn import svm
        # use the same dataset
        
        clf = svm.OneClassSVM(nu=0.05, kernel="rbf", gamma=0.1)
        clf.fit(X)
        
        svm.OneClassSVM(cache_size=200, coef0=0.0, degree=3, gamma=0.1, kernel='rbf',
              max_iter=-1, nu=0.05, random_state=None, shrinking=True, tol=0.001,
              verbose=False)
        
        osvm = clf.predict(X)
        
        # inliers are labeled 1, outliers are labeled -1
        normal = X[osvm == 1]
        abnormal = X[osvm == -1]
        
        '''---------------------IForest--------------------------'''
        from sklearn.ensemble import IsolationForest
        data = pd.DataFrame(X,columns = ["Price", "Time"])
        # train isolation forest
        model =  IsolationForest(contamination=0.1)
        model.fit(data) 
        data['IForest'] = pd.Series(model.predict(data))
        
        # visualization
        
        '''---------------------KNN--------------------------'''
        # train kNN detector
        from pyod.models.knn import KNN
        clf_name = 'KNN'
        clf = KNN()
        clf.fit(X)
        # get the prediction labels and outlier scores of the training data
        ss = clf.labels_  # binary labels (0: inliers, 1: outliers)
        #y_train_scores = clf.decision_scores_  # raw outlier scores

        data['OSVM']=osvm
        data['KNN']=ss
        
        # Convert each value of Knn to be equal with they others
        data.loc[(data.KNN == 0) , 'KNN'] = '1'  
        data.loc[(data.KNN == 1) , 'KNN'] = '-1'  
        
        
        #
        data['KNN']=data['KNN'].astype(int)
        data['OSVM']=data['OSVM'].astype(int)
        data['IForest']=data['IForest'].astype(int)
        #data['RES']=data['RES'].astype(int)
        
        #
        data['RES']=data.OSVM +data.IForest+data.KNN
        data.dtypes
        # Operation to get RES
        data.RES[data.RES == 1] = 0
        data.RES[data.RES == 3] = 1
        data.RES[data.RES == -3] = 0
        data.RES[data.RES == -1] = 0
        

        x=data.iloc[:,[0,1]].values
        y=data.iloc[:,[5]].values
        self.X_train,self.X_test,self.y_train,self.y_test = train_test_split(x, y, test_size = 0.3, random_state = 100) 
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_using_gini(self,X_train, y_train): 
  #criterion = "gini",contamination=float(.12),random_state = 100, min_samples_leaf=5
        # Creating the classifier object 
        self.model = DecisionTreeClassifier() 
      
        # Performing training 
        self.model.fit(self.X_train,self.y_train) 
        return self.model 
    
    def prediction(self,X_test, clf_object): 
      
        # Predicton on test with giniIndex 
        self.y_pred = clf_object.predict(self.X_test) 
        print("Predicted values:") 
        print(self.y_pred) 
        return self.y_pred 
    
    def cal_accuracy(self,y_test, y_pred): 
          
        print("Confusion Matrix: ", 
            confusion_matrix(self.y_test,self.y_pred)) 
          
        print ("Accuracy : ", 
        accuracy_score(self.y_test,self.y_pred)*100) 
          
        print("Report : ", 
        classification_report(self.y_test,self.y_pred)) 
    
        
