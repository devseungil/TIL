# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

#

from sklearn.datasets import load_wine

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

wine = load_wine(as_frame=True)

x = wine.data
y = wine.target

y = y.replace({0 : 'class_0' , 1 : 'class_1' , 2 : 'class_2'})

x_train , x_test , y_train , y_test = train_test_split(x,y , train_size=0.67  , random_state=0)

train = pd.concat([x_train , y_train] , axis=1)
test= pd.concat([x_test , y_test] , axis=1)

train['target']

# ## EDA : Explanatory Data Analysis = 탐색적 데이터 분석
#
# 목적 : 분류가 가능한지, 아닌지를 판단 목적없이 전체데이터의 평균과 분산을 계산하는건 의미없음
#

train.groupby('target')[['flavanoids','proline','color_intensity']].mean()

import seaborn as sns 

sns.boxplot(x='target' , y='proline' , data=train)

sns.boxplot(x='target' , y='flavanoids' , data=train)

sns.boxplot(x='target' , y='color_intensity' , data=train)

sns.scatterplot(x='flavanoids' , y='color_intensity' , hue='target' , data=train)
#클래스끼리 잘 분류돼있음 정규화나 표준화는 필요없어보임

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

param_grid = {'max_depth' : [3,4,5] , 'n_estimators' : [300,400,500] , 'criterion' : ["gini", "entropy"]}
clf = GridSearchCV(estimator = RandomForestClassifier(random_state=0) , param_grid=param_grid
                  , scoring='accuracy' , cv=5 , n_jobs= -1)
clf.fit(x_train , y_train)

print(clf.best_estimator_)
print(clf.best_params_)

clf_pred = clf.best_estimator_.predict(x_test)

print(classification_report(clf_pred , y_test))


