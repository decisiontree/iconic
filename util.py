# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 08:50:48 2017

@author: nplia
"""
import pandas as pd
import numpy as np

def gini(actual, pred):
    actual = np.asarray(actual) #In case, someone passes Series or list
    n = len(actual)
    a_s = actual[np.argsort(pred)]
    a_c = a_s.cumsum()
    giniSum = a_c.sum() / a_s.sum() - (n + 1) / 2.0
    return giniSum / n
 
def gini_norm(actual, pred):
    if pred.ndim == 2:#Required for sklearn wrapper
        pred = pred[:,1] #If proba array contains proba for both 0 and 1 classes, just pick class 1
    return gini(actual, pred) / gini(actual, actual)

def gini_weighted(actual, pred, weight):
    df = pd.DataFrame({"actual":actual,"pred":pred,"weight":weight}) 
    df = df.sort_values(by=['pred'],ascending=False).reset_index(drop=True)
    df["random"] = (df.weight / df.weight.sum()).cumsum()
    total_pos = (df.actual * df.weight).sum()
    df["cum_pos_found"] = (df.actual * df.weight).cumsum()
    df["lorentz"] = df.cum_pos_found / total_pos
    gini = sum(df.lorentz[1:].values * (df.random[:-1])) - sum(df.lorentz[:-1].values * (df.random[1:]))
    return gini

def gini_norm_weighted(actual, pred, weight):
    return gini_weighted(actual,pred,weight) / gini_weighted(actual,actual,weight)

def gini_lgb(pred, dtrain):
    label = dtrain.get_label()
    score = gini_norm(label, pred)
    return 'gini', score, True

def gini_lgb_weight(pred, dtrain):
    label = dtrain.get_label()
    weight = dtrain.get_weight()
    score = gini_norm_weighted(label, pred, weight)
    return 'gini', score, True