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

# # knn 분류 , knn 회귀 모형 비교를 해서 예측 모델을 생성하자
#
# 회귀 : neighbors.KNeighborsRegressor
#  - k개의 인접한 자료의 가중(평균)으로 예측
#  - 아래와 동일
#
# 분류 : neighbors.KNeighborsClassifier
#  - 새로운 값은 기존의 데이터를 기준으로 가까운 k개의 최근접 값을 기준으로 분류
#  - k값 짝수는 피하는게 좋음
#  - k가 1에 가까울수록 과적합 클수록 과소적합
#  - weights = uniform : 균일한가중치(디폴트) , distance : 거리에대한 가중치
#  - n_jobs(검색작업수) = -1 : cpu코어수가 설정됨
#

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error


df = pd.read_csv('diabetes.csv')

# x = 임신횟수 , 혈당, 혈압 y 아웃컴

df_train , df_test = train_test_split(df , train_size=0.8 , random_state=123)
df_train[:1]
x = df_train[['Pregnancies','Glucose','BloodPressure']]
y = df_train['Outcome']
x_test = df_test[['Pregnancies','Glucose','BloodPressure']]
y_test = df_test['Outcome']

df_model = KNeighborsClassifier().fit(x , y)

df_predict = df_model.predict(x_test)

print(classification_report(df_predict , y_test))

accuracy_score(df_predict , y_test)

# x = 임신경험유무 혈당 혈압 인슐린 체질량
#
# y = 아웃컴(당뇨발생유무)

df['is_preg'] = (df['Pregnancies'] > 0) + 0

df_train , df_test = train_test_split(df , train_size=0.8 , random_state=123)
df_train[:1]

for k in range(3,21,2) :
    model_2 = KNeighborsClassifier(n_neighbors=k).fit(df_train[['is_preg','Glucose','BloodPressure','Insulin','BMI']],df_train['Outcome'])
    predict_2 = model_2.predict(df_test[['is_preg','Glucose','BloodPressure','Insulin','BMI']])
    print("k가",k,"일때", "정확도:",accuracy_score(df_test['Outcome'] , predict_2).round(3))

# x = 임신경험유무 혈당 혈압 인슐린
#
# y = 체질량(수치는 회귀)

for k in range(3,21,2) :
    model_3 = KNeighborsRegressor(n_neighbors=k).fit(df_train[['is_preg','Glucose','BloodPressure','Insulin']],df_train['BMI'])
    predict_3 = model_3.predict(df_test[['is_preg','Glucose','BloodPressure','Insulin']])
    print(k,(mean_squared_error(predict_3 , df_test['BMI'])**0.5).round(3))

from sklearn.datasets import load_iris

iris = load_iris()

x = iris.data
y = iris.target
x_train , x_test , y_train , y_test = train_test_split(x,y,random_state=0)

# +
lst_n = []
lst_score = []
for k in range(1,31) :
    model_iris = KNeighborsClassifier(n_neighbors=k).fit(x_train , y_train)
    pre_iris = model_iris.predict(x_test)
    score = model_iris.score(x_test , y_test)#스코어는 회귀모델 :r2(결정계수값) /분류모델 :accuracy(정확도)값
    lst_n.append(k)
    lst_score.append(score)
    
print(classification_report(pre_iris , y_test))
# -

plt.ylim(0.85,1.0)
plt.xlabel('n_neighbors')
plt.ylabel('score')
plt.plot(lst_n , lst_score)

from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer(as_frame=True)

x= cancer.data
y= cancer.target

x_trainc , x_testc , y_trainc , y_testc = train_test_split(x,y,random_state=0)

for k in range(1,26) :
    model_cancer = KNeighborsClassifier(n_neighbors=k).fit(x_trainc,y_trainc)
    pred_cancer = model_cancer.predict(x_testc)
    print(k , model_cancer.score(x_testc, y_testc).round(3))


