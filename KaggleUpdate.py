#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 18:09:41 2019

@author: balmeet
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import xgboost as xgb
from scipy import stats
import category_encoders as cat
from sklearn.preprocessing import OneHotEncoder
#columns = ['Instance','YearOfRecord','Gender','Age','Country','SizeOfCity','Profession',
#           'UniversityDegree','WearGlasses','HairColor','BodyHeight','IncomeEUR']
trainCol = pd.read_csv('tcd ml 2019-20 income prediction training (with labels).csv')
testCol = pd.read_csv('tcd ml 2019-20 income prediction test (without labels).csv')
#testOut = pd.read_csv('tcd ml 2019-20 income prediction submission file.csv') 

#trainCol = trainCol.drop(['Size of City','Wears Glasses','Body Height [cm]','Instance'],axis = 1)

    
x_mod = trainCol
#import seaborn as sns
#sns.boxplot(x=x_mod['Income in EUR'])

#x_mod = x_mod[x_mod['Income in EUR'] < 2500000]

#import seaborn as sns
#sns.boxplot(x=x_mod['Gender'])

x_mod['Year of Record'].dropna(x_mod['Year of Record'].mean(), inplace = True)
x_mod['Age'].fillna(x_mod['Age'].median(), inplace = True)
#x_mod['Gender'].fillna(method = 'ffill', inplace = True)
#x_mod['Size of City'].fillna(method = 'ffill', inplace = True)
#x_mod['Wears Glasses'].fillna(method = 'ffill', inplace = True)
#x_mod['Body Height [cm]'].fillna(method = 'ffill', inplace = True)
#x_mod['University Degree'].fillna(method = 'ffill', inplace = True)
#x_mod['Hair Color'].fillna(method = 'ffill', inplace = True)
#x_mod['Country'].fillna(method = 'ffill', inplace = True)
#x_mod['Profession'].fillna(method = 'ffill', inplace = True)

#x_mod.astype({'Gender': 'category'}).dtypes
#x_mod.astype({'University Degree': 'category'}).dtypes
#x_mod.astype({'Hair Color': 'category'}).dtypes
#x_mod.astype({'Profession': 'category'}).dtypes
#x_mod.astype({'Country': 'category'}).dtypes

x_mod.fillna({"Gender": "missing", "University Degree":"missing","Hair Color": "missing", 'Profession':'missing' })
"""
def clean_gender(gen):
    if gen == '0' or gen == 0:
        gen = 'unknown'
    return gen

x_mod['Gender'] = x_mod['Gender'].apply(clean_gender)
x_mod['Gender'].value_counts()


def clean_haircolor(hc):
    if hc == '0' or hc == 0 or hc == 'Unknown':
        hc = 'Red'
    return hc

x_mod['Hair Color'] = x_mod['Hair Color'].apply(clean_haircolor)
x_mod['Hair Color'].value_counts()

def clean_UD(ud):
    if ud == '0' or ud == 0:
        ud = 'PhD'
    return ud

x_mod['University Degree'] = x_mod['University Degree'].apply(clean_UD)
x_mod['University Degree'].value_counts()
"""
catCol1 = ['Profession','Country']

encode = cat.BinaryEncoder(cols = catCol1)
x_mod = encode.fit_transform(x_mod)

catCol2 = ['Gender','University Degree','Hair Color']

encode02 = cat.BackwardDifferenceEncoder(cols = catCol2)
x_mod = encode02.fit_transform(x_mod)


"""
def ExtractingSplFeatures(uniques,dataFrame,columnName):
    
    # OneHeartEncoder
    encoder = OneHotEncoder(categories = [uniques],sparse = False, handle_unknown = 'ignore')

    #reshape the column
    column = dataFrame[columnName]
    column = np.array(column).reshape(-1,1)

    #Extract the column and join the data frame
    dataFrame = dataFrame.join(pd.DataFrame(encoder.fit_transform(column),columns=encoder.categories_,index=dataFrame.index))

    #Remove the profession Column
    dataFrame = dataFrame.drop([columnName], axis = 1)

    return dataFrame

prof = x_mod['Profession'].unique()
cont = x_mod['Country'].unique()

x_mod = ExtractingSplFeatures(prof,x_mod,'Profession')
x_mod = ExtractingSplFeatures(cont,x_mod,'Country')

"""
"""
dummy = pd.get_dummies(x_mod['University Degree'])
x_mod = pd.concat([x_mod,dummy],axis = 1)
x_mod = x_mod.drop(['University Degree'],axis = 1)

dummy = pd.get_dummies(x_mod['Hair Color'])
x_mod = pd.concat([x_mod,dummy],axis = 1)
x_mod = x_mod.drop(['Hair Color'],axis = 1)

x_mod['Country'] = x_mod['Country'].astype('category').cat.codes
x_mod['Profession'] = x_mod['Profession'].astype('category').cat.codes
x_mod['Gender'] = x_mod['Gender'].astype('category').cat.codes

dummy = pd.get_dummies(x_mod['Gender'])
x_mod = pd.concat([x_mod,dummy],axis = 1)
x_mod = x_mod.drop(['Gender'],axis = 1)
"""

#quant1 = x_mod['Year of Record'].quantile(0.90)
#x_mod = x_mod[x_mod['Year of Record'] < quant1]
#x_mod.shape

y_mod = x_mod.iloc[0:,-1]
x_mod = x_mod.drop(columns = 'Income in EUR')

#x_train, x_test, y_train, y_test = train_test_split(x_mod, y_mod, test_size = 0.2)

#reg = LinearRegression()
reg2 = BayesianRidge()
#reg2.fit(x_train, y_train)
#prediction = reg2.predict(x_test)

#reg.fit(x_mod,y_mod)
reg2.fit(x_mod,y_mod)
"""
from sklearn.metrics import mean_squared_error
from math import sqrt

rms = sqrt(mean_squared_error(y_test, prediction))
rms
"""
### Test File Below ###

#testCol = testCol.drop(['Size of City','Wears Glasses'],axis = 1)
x_t = testCol

x_t['Year of Record'].fillna(x_t['Year of Record'].mean(), inplace = True)
x_t['Age'].fillna(x_t['Age'].median(), inplace = True)
#x_t['Gender'].fillna(method = 'ffill', inplace = True)
#x_t['Country'].fillna(method = 'ffill', inplace = True)
#x_t['Profession'].fillna(method = 'ffill', inplace = True)
#x_t['Body Height [cm]'].fillna(method = 'ffill', inplace = True)
#x_t['University Degree'].fillna(method = 'ffill', inplace = True)
#x_t['Hair Color'].fillna(method = 'ffill', inplace = True)
#x_t['Country'].fillna(method = 'ffill', inplace = True)
#x_t['Profession'].fillna(method = 'ffill', inplace = True)

#x_t.astype({'Gender': 'category'}).dtypes
#x_t.astype({'University Degree': 'category'}).dtypes
#x_t.astype({'Hair Color': 'category'}).dtypes
#x_t.astype({'Profession': 'category'}).dtypes
#x_t.astype({'Country': 'category'}).dtypes


x_t.fillna({"Gender": "missing", "University Degree":"missing","Hair Color": "missing", 'Profession':'missing' })

"""
def clean_gendertest(gent):
    if gent == '0' or gent == 0:
        gent = 'unknown'
    return gent

x_t['Gender'] = x_t['Gender'].apply(clean_gendertest)
x_t['Gender'].value_counts()


def clean_haircolortest(hct):
    if hct == '0' or hct == 0 or hct == 'Unknown':
        hct = 'Red'
    return hct

x_t['Hair Color'] = x_t['Hair Color'].apply(clean_haircolortest)
x_t['Hair Color'].value_counts()

def clean_UDtest(udt):
    if udt == '0' or udt == 0:
        udt = 'PhD'
    return udt

x_t['University Degree'] = x_t['University Degree'].apply(clean_UD)
x_t['University Degree'].value_counts()
"""

#catCol1 = ['Profession','Country']

encodeTest1 = cat.BinaryEncoder(cols = catCol1)
x_t = encodeTest1.fit_transform(x_t)

#catColTest = ['Gender','University Degree','Profession','Country','Hair Color']

encodeTest2 = cat.BackwardDifferenceEncoder(cols = catCol2)
x_t = encodeTest2.fit_transform(x_t)

#profTest = x_t['Profession'].unique()
#contTest = x_t['Country'].unique()

#x_t = ExtractingSplFeatures(prof,x_t,'Profession')
#x_t = ExtractingSplFeatures(cont,x_t,'Country')

#quant1_test = x_t['Year of Record'].quantile(0.90)
#x_t = x_t[x_t['Year of Record'] < quant1_test]
#x_t.shape
#testInstance = x_t['Instance']
#testCol['Income']

x_t = x_t.drop(columns = 'Income')


y_testpred = reg2.predict(x_t)
testCol['Income'] = y_testpred
"""
outputDataFrame = pd.DataFrame(testInstance)
outputDataFrame['Income'] = pd.DataFrame(y_testpred)
print("pushing to a file") 
outputDataFrame.to_csv('myPrediction_06.csv')
"""
testCol.to_csv('myPrediction_07.csv')

y_testpred.shape
x_t.columns
y_t.columns
testCol.columns
