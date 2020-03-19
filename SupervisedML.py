#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 17:03:28 2020

@author: christopher
"""

# Imports

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF
import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as ff

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from catboost import CatBoostClassifier
from sklearn.metrics import auc,confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import log_loss

# Dataset
df = pd.read_csv('Churn_Modelling.csv')

# Data Cleaning

df.columns = df.columns.str.strip('')
df.rename(columns={'Exited': 'Churn'}, inplace =True)

replace_bool_cols = ['HasCrCard', 'IsActiveMember']
for i in replace_bool_cols:
    df[i] = df[i].replace({1:'Yes', 0:'No'})
    
df.NumOfProducts = df.NumOfProducts.replace({1:'One', 2:'Two', 3:'Three', 4:'Four'})

# Categorize tenure
def tenure_cat(df):
    if df.Tenure <= 2:
        return 'Tenure 0-2 Years'
    elif (df.Tenure > 2) & (df.Tenure <= 4):
        return 'Tenure 3-4 Years'
    elif (df.Tenure > 4) & (df.Tenure <= 6):
        return 'Tenure 5-6 Years'
    elif df.Tenure >7:
        return 'Tenure 7+ Years'
df['TenureGroup'] = df.apply(lambda df: tenure_cat(df), axis=1)



df.drop(['RowNumber', 'Surname', 'CustomerId', 'Tenure', 'CreditScore'], axis=1, inplace=True)

# Categorize Credit Score Bad, Fair, Good, Excellent
def credit_cat(df):
    if df.CreditScore <= 629:
        return 'Bad'
    elif (df.CreditScore > 630) & (df.CreditScore <= 689):
        return 'Fair'
    elif (df.CreditScore > 690) & (df.CreditScore <= 719):
        return 'Good'
    elif df.CreditScore > 720:
        return 'Excellent'
df['CreditScoreGroup'] = df.apply(lambda df: credit_cat(df), axis=1)

df.drop(['RowNumber', 'Surname', 'CustomerId', 'Tenure', 'CreditScore'], axis=1, inplace=True)

# Making a list of categorical and numerical columns
target = ['Churn']
cat_columns = df.nunique()[df.nunique() < 5].keys().tolist()
cat_columns = [i for i in cat_columns if i not in target]
num_columns = [i for i in df.columns if i not in cat_columns + target]


# Exploratory Analysis

churn_pie = plt.pie(df.Churn.value_counts(), startangle=0, autopct='%5.0f%%', pctdistance=0.5, radius=1.2, explode=(0.1,0.0))
labels=df.Churn.unique()
plt.title('Customer Churn Percentage in Data', weight='bold', size=14)
plt.legend(churn_pie,labels = ['No', 'Yes'], bbox_to_anchor=(0.9,0.7), fontsize=10, 
           bbox_transform=plt.gcf().transFigure)
plt.subplots_adjust(left=0.0, bottom=0.1, right=0.85)
plt.rcParams.update({'font.size': 15})
plt.show()
plt.savefig('CustomerChurnPercentage', dpi=600)

def mean_churn_by_cat(ColumnName):
    pd.crosstab(df.Churn,df[i], normalize='index').plot.bar(rot=0)
    plt.title(f'Percent Churn By {i}', weight='bold', size=14)
    plt.legend(loc='best')
    plt.savefig(f'Percent Churn by {i}', dpi=600)
    return plt.show()

for i in cat_columns:
    mean_churn_by_cat(i)
    
# Data Normalization
    
def standardize_num_cols(df):
    scaler=MinMaxScaler()
    df = scaler.fit_transform(df.values.reshape(-1, 1))
    return df

for i in num_columns:
    standardize_num_cols(df[i])
    
scaler=MinMaxScaler()
df.Age = scaler.fit_transform(df.Age.values.reshape(-1, 1))
df.Balance = scaler.fit_transform(df.Balance.values.reshape(-1, 1))
df.EstimatedSalary = scaler.fit_transform(df.EstimatedSalary.values.reshape(-1, 1))
    
# Make Dummies

df = pd.get_dummies(data = df,columns=cat_columns, drop_first=True) 

# VIF And Correlation Matrix

df_vif = df.drop(['Churn'], axis=1)
pd.Series([VIF(df_vif.values, i) for i in range(df_vif.shape[1])],index=df_vif.columns).sort_values(ascending=False)

# VIF Correlation Visualization

correlation = df.corr()
#tick labels
matrix_cols = correlation.columns.tolist()
#convert to array
corr_array  = np.array(correlation)

#Plotting
trace = go.Heatmap(z = corr_array,
                   x = matrix_cols,
                   y = matrix_cols,
                   colorscale = "Viridis",
                   colorbar   = dict(title = "Pearson Correlation coefficient",
                                     titleside = "right"
                                    ) ,
                  )

layout = go.Layout(dict(title = "Correlation Matrix for variables",
                        autosize = False,
                        height  = 720,
                        width   = 800,
                        margin  = dict(r = 0 ,l = 210,
                                       t = 25,b = 210,
                                      ),
                        yaxis   = dict(tickfont = dict(size = 9)),
                        xaxis   = dict(tickfont = dict(size = 9))
                       )
                  )

data = [trace]
fig = go.Figure(data=data,layout=layout)
py.iplot(fig)

# Machine Model
    
# Train/Test Split for 33% of the data
X_train, X_test, y_train, y_test = train_test_split(df.drop('Churn', axis=1), df.Churn, test_size=0.33)
# Initialize data for catboost
dummies = ['Geography', 'Gender', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'TenureGroup', 'CreditScoreGroup']
cat_features = dummies
# Initialize CatBoostClassifier
model = CatBoostClassifier(iterations=500, # more iterations to learn from
                          eval_metric='Recall') # optimize Recall to reduce false negatives

model.fit(X_train, y_train, cat_features)
# Get predicted classes
preds_class = model.predict(X_test)   
    
# Model Metrics

confusion_matrix(y_test, preds_class)

model_CB_roc = roc_auc_score(y_test,preds_class)

fpr,tpr,thresholds = roc_curve(y_test, model.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr,tpr,label=f'Model CatBoost (area={model_CB_roc})')
plt.plot([0,1], [0,1])
plt.legend()
plt.savefig('Catboost Model', dpi=600)
    
accuracy_score(y_test,preds_class)
recall_score(y_test,preds_class)
precision_score(y_test,preds_class)
f1_score(y_test,preds_class)
log_loss(y_test, preds_class)
