#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  4 17:44:59 2019

@author: has
"""

import Alertifier as Al

  
# Void main
ag=Al.Alertifier('data.pkl','KV')
ag.filters()
ag.GetLabel(ag.X)


#ag.cal_accuracy(y_test,y_pred_gini)



if ag.model!=None:
    print('the mode is deployed')    
    ag.prediction(ag.X_test,ag.model) 
else:
    print('model not deployed')
    ag.train_using_gini(ag.X_train,ag.y_train)
    ag.prediction(ag.X_test,ag.model) 




#print(clf_gini)

#X_train, X_test, y_train, y_test=ag.splitdataset(lab)

#lab=ag.GetLabel(X)
#X_train, X_test, y_train, y_test=splitdataset(Xlab_)




    