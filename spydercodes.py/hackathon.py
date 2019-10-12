# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 00:09:24 2019

@author: USER
"""

import pandas as pd
from sklearn import preprocessing
data= pd.read_csv('‪C:\\Users\\USER\\Desktop\\first.csv')
import matplotlib.pyplot as plt

df.dropna(subset=['Loan Status'], inplace = True)
le = preprocessing.LabelEncoder()
df['Loan Status'] = le.fit_transform(df['Loan Status'])

q=df.corr()['Loan Status'].sort_values(ascending=False)

null_columns=df.columns[df.isnull().any()]
df[null_columns].isnull().sum()

# label encoding of loan status
df.dropna(subset=['Loan Status'], inplace = True)
le = preprocessing.LabelEncoder()
df['Loan Status'] = le.fit_transform(df['Loan Status'])
df['Term'].replace(("Short Term","Long Term"),(0,1), inplace=True)
df.head()


# making one hot encoding
'''
df['Credit Score'] = df['Credit Score'].fillna(df['Credit Score'].mode()[0])
import numpy as np
df['Credit Score'] = df['Credit Score'].apply(lambda val: "0" if np.isreal(val) and val < 580 else val)
df['Credit Score'] = df['Credit Score'].apply(lambda val: "1" if np.isreal(val) and (val >= 580 and val < 670) else val)
df['Credit Score'] = df['Credit Score'].apply(lambda val: "2" if np.isreal(val) and (val >= 670 and val < 740) else val)
df['Credit Score'] = df['Credit Score'].apply(lambda val: "3" if np.isreal(val) and (val >= 740 and val < 800) else val)
df['Credit Score'] = df['Credit Score'].apply(lambda val: "4" if np.isreal(val) and (val >= 800 and val <= 850) else val)
'''
'''
q=len(df['Years in current job'])
for i in  range(q):
    if df['Years in current job'][i].isna() and df['Annual Income'][i].isna() :
        print("1")
'''

df['Years in current job']=df['Years in current job'].map({'8 years':8, '10+ years':15, '3 years':3, '5 years':5, '< 1 year':0.5, '2 years':2, '4 years':4, '9 years':9, '7 years':7, '1 year':1, '6 years':6})
df['Credit Score']=df['Credit Score'].apply(lambda v:(v/10) if v>850 else v)
df['Credit Score'].describe()


m = df[df['Years in current job'].isna()].index
q = df[df['Annual Income'].isna()].index
q=list(q)
m=list(m)
for i in m:
    for j in q:
       if j==i:
        df['Years in current job'][j]=0
        df['Annual Income'][j]=0
df['Annual Income'] = df['Annual Income'].fillna((df['Annual Income'].mean()))
df['Years in current job'].fillna(0, inplace=True)
df['Maximum Open Credit'] = df['Maximum Open Credit'].fillna(df['Maximum Open Credit'].mode()[0])

df.shape
# converting to csv (from dataframe)
# df.to_csv('C:\\Users\\USER\\Desktop\\newdf.csv',sep=',')
#e=df.corr()
df['Credit Score'] = df['Credit Score'].fillna(df['Credit Score'].mode()[0])

# finding outlier
data_mean, data_std =df['Current Loan Amount'].mean(),df['Current Loan Amount'].std()
# identify outliers
cut_off = data_std * 3
lower, upper = data_mean - cut_off, data_mean + cut_off
# identify outliers
outliers = [x for x in df['Current Loan Amount'] if x < lower or x > upper]
# remove outliers
outliers_removed = [x for x in df['Current Loan Amount']  if x > lower and x < upper]
# up  sampling
from sklearn.model_selection import train_test_split
df['Loan Status'].value_counts()
	
from sklearn.utils import resample
df_majority = df['Loan Status']==1
df_minority = df['Loan Status']==0

df_minority_upsampled = resample(df_minority, replace=True,n_samples=100000,random_state=123)
df_upsampled = pd.concat([df_majority, df_minority_upsampled])
df_upsampled.balance.value_counts()
import matplotlib.pyplot as plt
def get_feature_groups():
    """ Returns a list of numerical and categorical features,
    excluding SalePrice and Id. """
    # Numerical Features
    num_features = df.select_dtypes(include=['int64','float64']).columns
     # drop ID and SalePrice
    # drop ID and SalePrice

    # Categorical Features
    cat_features = df.select_dtypes(include=['object']).columns
    return list(num_features), list(cat_features)



num_features, cat_features = get_feature_groups()



y = dataframe['Loan Status']
X = dataframe.drop(['Loan Status'],axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
knnclassifier = KNeighborsClassifier(n_neighbors = int(X.shape[1]/2))
knnclassifier.fit(X_train, y_train)
prediction = knnclassifier.predict(X_test)
print("Accuracy Score: ", accuracy_score(y_test, prediction))

# lR
lregclassifier = LogisticRegression()
lregclassifier.fit(X_train,y_train)
lregprediction = lregclassifier.predict(X_test)
print("Score: ",lregclassifier.score(X_test, y_test))

# svm
from sklearn.svm import SVC
clf = SVC(gamma='auto', kernel ='linear')
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
print("Accuracy Score: ", accuracy_score(y_test, pred))






    