# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 16:44:18 2017

@author: nzcga
"""
from sklearn.base import TransformerMixin
#from sklearn.pipeline import Pipeline, FeatureUnion
#from pandas import DataFrame
from statistics import median
import pandas as pd
import warnings
import numpy as np
from sklearn.pipeline import Pipeline

# Constant Variables
imput_appendix = '___imp'
imput_flag_appendix = '___mflg'
imput_missing_cat = '_missing'

DISPLAY_WIDTH = 50


class ColumnConf:
    def  __init__(self, config_row):
        self.config_row = config_row
        self.name = config_row["Column"]
        self.inputType = config_row["inputType"]
        self.input = config_row["input"]
        self.imputeType = config_row["imputeType"]
    def __str__(self):
        return '[ColumnConf Object: '+self.name+', ' +self.inputType +']'
    def __repr__(self):
        return self.__str__(self) 


class ColumnConfigs:
    def  __init__(self, config_df):
        self.config_df = config_df
        self.columnDict = {}
        for index, row in config_df.iterrows():
            newColumn = ColumnConf(row)
            self.columnDict[newColumn.name] = newColumn
    
    def getImputer(self, column):
        try:
            imputeType = self.columnDict.get(column).imputeType
        except:
            imputeType = '_na_'
        #print('impute type is: '+imputeType)
        return imputeType
    def getDType(self, column):
        try:
            dType = self.columnDict.get(column).inputType
        except:
            dType = '_na_'
        #print('impute type is: '+imputeType)
        return dType
    def isInput(self, column):
        try:
            inputFlag = self.columnDict.get(column).input
        except:
            inputFlag = '_na_'
        
        if(inputFlag == 'input'):
            return True
        elif(inputFlag == '_na_'):
            warnings.warn("[Missing Config - " + column + "]"  )
            return False
        else:
            return False




class VarRemover(TransformerMixin):
    def __init__(self, config):
        self.config = config
        self.inputFlags = {}
        self.dropColumns = []
    
    def fit(self, X, y=None):
        for column in X:
            inputFlag = self.config.isInput(column)
            #print('find input flag '+ str(inputFlag)+ ' for column - ' + column)
            if(X.shape[0]==X[column].isnull().sum()):
            #if(X[column].dropna().nunique() <= 0): # only keep it if has value
                inputFlag = False
                warnings.warn('[DROPPING - '+column+'] - pure-missing value')
            self.inputFlags[column] = inputFlag
            if (inputFlag==False):
                self.dropColumns.append(column)
        return self
        
    def transform(self, X, y=None):
        X = X.drop(self.dropColumns,inplace=False,axis=1,errors='ignore')
        print('[Variable Dropped]')
        return X





class AllModelImputer(TransformerMixin):
    def __init__(self, config):
        self.config = config
        self.imputeValues = {} #imputation values
        self.imputeColumns = [] #columns with missing
        self.imputeOriCols = {} #original column names for missing flags
    
    def fit(self, X, y=None):
        print('[Detecting Imputation]:')
        
        self.imputeColumns = X.columns[X.isnull().any()]
        #self.imputeColumns = X.columns
        
        #reg ori column names
        for column in X:
            self.imputeOriCols[column+imput_flag_appendix]=column
        
        i=1
        ori_cols=X.shape[1]
        for column in X:
            imputeType = self.config.getImputer(column)
            #print('[Impute type '+ imputeType+ '] - ' + column)
            if imputeType == 'zero': #impute to 0
                self.imputeValues[column] = 0.0
                print('0', end='')
            elif imputeType == 'median': #median value
                self.imputeValues[column] = median(X[column].dropna())
                print('-', end='')
            elif imputeType == 'missing': #impute to missing string, based on constant imput_missing_cat
                self.imputeValues[column] = imput_missing_cat
                print('.', end='')
            elif imputeType == 'no': #get the first value on 0/1, n/y, no/yes
                self.imputeValues[column] = sorted(X[column].dropna().unique())[0] 
                print('x', end='')
            elif imputeType == 'min': #get the min value
                self.imputeValues[column] = sorted(X[column].dropna().unique())[0] 
                print('_', end='')
            elif imputeType == 'max': #get the max value
                self.imputeValues[column] = sorted(X[column].dropna().unique(),reverse=True)[0]
                print('+', end='')
            elif imputeType == 'pcntl': #get the max Pcntl
                self.imputeValues[column] = 101
                print('1', end='')
            else: #default is median, works for both categorical and numeric variables
                self.imputeValues[column] = median(X[column].dropna())
                print('!', end='')
            
            if(i % DISPLAY_WIDTH == 0):
                print("  |{0:5d} ({1:+3.2f}%)".format(i,100.0*i/ori_cols)) #new line after 100 chars
            i=i+1
        print("  |{0} (100.00 %)".format(i))
        return self
        
    def transform(self, X, y=None):
        #new_X = []
        #for column in X:
            #new_X.append(X[column].fillna(self.imputeValues[column]))
        print('[Imputation Start]:')
        colWithNaN = X.columns[X.isnull().any()]
        
        i=1
        ori_cols=X.shape[1]
        for column in X:
            if(column in self.imputeColumns):
                #create a Missing Flag for the imputation
                X[column+imput_flag_appendix]=X[column].isnull()*1
            if(column in self.imputeValues and column in colWithNaN):
                X[column]=X[column].fillna(self.imputeValues[column], inplace=False)
                print('=', end='')
            else:
                print(' ', end='')
                #pass
                
            #print(new_X.columns)
            
            if(i % DISPLAY_WIDTH == 0):
                 print("  |{0:5d} ({1:+3.2f}%)".format(i,100.0*i/ori_cols)) #new line after 100 chars
            i=i+1
        print("  |{0} (100.00 %)".format(i))
        return X

class ToLowerCase(TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
        
    def transform(self, X, y=None):
        for column in X:
            #print(column)
            if(X.dtypes[column] == np.object):
                X[column]=X[column].astype(str).str.lower()
            else:
                pass
            
            #print(new_X.columns)
        print('[Lower Case Conversion Done]:')
        return X




class ConvertDType(TransformerMixin):
    def __init__(self, config):
        self.config = config
        self.dtypes = {}
        self.numericColumns = []
        self.categoryColumns = []
    
    def fit(self, X, y=None):
        for column in X:
            dtype = self.config.getDType(column)
            self.dtypes[column] = dtype
            if (dtype=='numeric' and (not np.issubdtype(X.dtypes[column], np.number))):
                #print('Detect dtype change to numeric: ['+ dtype+ '] for column - ' + column)
                self.numericColumns.append(column)
            if (dtype!='numeric' and (np.issubdtype(X.dtypes[column], np.number))):
                #print('Detect dtype change to string: ['+ dtype+ '] for column - ' + column)
                self.categoryColumns.append(column)
        print('[Convert Found] - {0} columns to numeric. {1} columns to category.'.format(len(self.numericColumns),len(self.categoryColumns) ))
        return self
        
    def transform(self, X, y=None):
        #X[self.numericColumns] = X[self.numericColumns].astype(float)
        numCols = list(set(X.columns).intersection(self.numericColumns))
        if(len(numCols) > 0):
            df_num = X[numCols].apply(pd.to_numeric, errors='coerce')
            #df_num = X[self.numericColumns].astype(float)
            X = X.drop(numCols,inplace=False,axis=1,errors='ignore')
            X = pd.concat([X, df_num], axis=1)
            print('[DType Convert Done] - {0} columns to numeric'.format(len(numCols) ))
        
        catCols = list(set(X.columns).intersection(self.categoryColumns))
        if(len(catCols) > 0):
            df_str = X[catCols].astype(str)
            X = X.drop(catCols,inplace=False,axis=1,errors='ignore')
            X = pd.concat([X, df_str], axis=1)
            print('[DType Convert Done] - {0} columns to category.'.format(len(catCols) ))
        return X

class DummyConverter(TransformerMixin):
    def __init__(self, config, max_level=48, drop='2-1'):
        self.config = config
        self.converts = {}
        self.dummies = {}
        self.maxLevel = max_level
        self.drop = drop #2-1, n-1 or n
        self.imputeOriCols = {} #original column names for dummy vars
    
    def fit(self, X, y=None):
        for column in X:
            dtype = self.config.getDType(column)
            if(dtype == 'category'):
                if(X[column].nunique() > self.maxLevel) : 
                    warnings.warn('[Dropping {0}] - More than {1} levels'.format(column,self.maxLevel ))
                    self.converts[column] = 'drop'
                
                elif(X[column].nunique()>0) :
                    #convert
                    self.converts[column] = 'convert'
                    dummyValues = X[column].unique()
                    self.dummies[column] = dummyValues
                else:
                    warnings.warn('[Dropping ' + column + '] - Unique Value')
                    self.converts[column] = 'drop'
                
                
            else:
                self.converts[column] = 'ignore'
        print('[Dummy Converion Detection Done]')
        
        #reg ori column names for dummy vars
        for column in X:
            if(self.converts[column]=='convert'):
                #print('Dummy Converting column:' + column)
                sizeOfDummy=len(self.dummies[column])
                if (self.drop=='2-1' and sizeOfDummy==2):
                    sizeOfDummy = sizeOfDummy - 1
                elif(self.drop=='n-1' and sizeOfDummy > 1):
                    sizeOfDummy = sizeOfDummy - 1
                else:
                    pass
                
                for idx in range(sizeOfDummy):
                    elem =  self.dummies[column][idx]
                #for elem in self.dummies[column]:
                    clean_elem = elem.translate ({ord(c): "_" for c in "!@#$%^&*()[]{};:,./<>?\|`~-=_+ "})
                    self.imputeOriCols[column+'__'+str(clean_elem)]=column

        
        return self
        
    def transform(self, X, y=None):
        dropColumns = []
        for column in X:
            if(self.converts[column]=='convert'):
                #print('Dummy Converting column:' + column)
                sizeOfDummy=len(self.dummies[column])
                if (self.drop=='2-1' and sizeOfDummy==2):
                    sizeOfDummy = sizeOfDummy - 1
                elif(self.drop=='n-1' and sizeOfDummy > 1):
                    sizeOfDummy = sizeOfDummy - 1
                else:
                    pass
                for idx in range(sizeOfDummy):
                    elem =  self.dummies[column][idx]
                #for elem in self.dummies[column]:
                    clean_elem = elem.translate ({ord(c): "_" for c in "!@#$%^&*()[]{};:,./<>?\|`~-=_+ "})
                    X[column+'__'+str(clean_elem)] = (X[column] == elem).apply(lambda x: int(x))
                print('[Hot Encode] - {0} -> {1} dummy vars'.format(column,sizeOfDummy))
                dropColumns.append(column)
            elif(self.converts[column]=='ignore'):
                pass
            elif(self.converts[column]=='drop'):
                dropColumns.append(column) #drop the column
            else: #drop column because of unique value
                pass
        if(len(dropColumns) > 0):
            
            X=X.drop(dropColumns, axis=1, inplace=False) 
            print('[Dummy Encode Done] - {0} columns converted'.format(len(dropColumns)))
        return X




class ModelDataProcessor():
    def __init__(self, config_file_path, max_level=48, drop='2-1'):
        self.config = ColumnConfigs(pd.read_csv(config_file_path, dtype=object))
        self.max_level = max_level
        self.drop = drop #2-1, n-1 or n
        self.ori_columns_dict = {}
        self.modelPreProcess = Pipeline([('remover', VarRemover(self.config)),
                                         ('conveter', ConvertDType(self.config)),
                                         ('imputer', AllModelImputer(self.config)),
                                         ('lower', ToLowerCase()),
                                         ('dummy', DummyConverter(self.config,max_level=self.max_level, drop=self.drop)),
                                         ])
    
    #first method need to be called before everything else
    def fit_transform(self, X, y=None):
        X_trans = self.modelPreProcess.fit_transform(X, y)
        self.ori_columns_dict = {}
        self.ori_columns_dict.update(self.modelPreProcess.steps[2][1].imputeOriCols)
        self.ori_columns_dict.update(self.modelPreProcess.steps[4][1].imputeOriCols)
        print(X_trans.shape)
        return X_trans
    
    def transform(self, X, model_var_list=None):
        if model_var_list is None:
            #all variables in
            X_trans = self.modelPreProcess.transform(X)
        else:
            #only use part of the model variables
            
            #step 1: transform to ori vari names
            var_ori_name = self.get_ori_columns(model_var_list)
            
            #Step 2: transform
            X_trans = self.modelPreProcess.transform(X[var_ori_name])[model_var_list]
        print(X_trans.shape)
        return X_trans
    
    def get_ori_columns(self, model_var_list):
        #transform model variables to ori vari names
        var_ori_name = []
        for col in model_var_list:
            if col in self.ori_columns_dict:
                var_ori_name.append(self.ori_columns_dict[col])
            else:
                var_ori_name.append(col)    
        return var_ori_name
    