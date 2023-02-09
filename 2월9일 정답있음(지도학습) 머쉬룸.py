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

from sklearn.datasets import load_wine
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier , AdaBoostClassifier , GradientBoostingClassifier , StackingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score , classification_report
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

# 데이터셋 로드 > 전처리 > EDA > 기계학습 예측모델작성 > 성능평가

mr = pd.read_csv('mushroom.csv' , header=None )
mr.columns =["edibility","cap-shape","cap-surface","cap-color","bruises","odor","gill-attachment","gill-spacing","gill-size","gill-color","stalk-shape","stalk-root","stalk-surface-above-ring","stalk-surface-below-ring","stalk-color-above-ring","stalk-color-below-ring","veil-type","veil-color","ring-number","ring-type","spore-print-color","population","habitat"]

# 누락값확인 -> 있으면 평균or중앙값으로대처 , 너무많으면 통쨰로 삭제
mr.isna().sum() 

#데이터를 가변화
# 순서에 관계가없다 => 더미변수 (다중공선성VIF 10이하의 경우 )
#관계있다 -> 카테고리변수화
mr_dummy = pd.get_dummies(mr, drop_first=True, 
     columns=["cap-shape","cap-surface","cap-color","bruises","odor","gill-attachment","gill-color","stalk-shape","stalk-root","stalk-surface-above-ring","stalk-surface-below-ring","stalk-color-above-ring","stalk-color-below-ring","veil-type","veil-color","ring-type","spore-print-color","habitat"])

#순서관계가 있는 특징은 카테고리변수화 -> 문자를 라벨 인코딩
mr_dummy

from sklearn import preprocessing
for col in ['edibility','gill-spacing','gill-size','ring-number','population'] :
    select_col = mr_dummy[col]
    en = preprocessing.LabelEncoder().fit(select_col)
    col_en = en.transform(select_col)
    mr_dummy[col] = col_en


mr_dummy.head()

x_train , x_test , y_train , y_test = train_test_split(mr_dummy.iloc[:,1:] , mr_dummy['edibility'] , train_size=0.7 , random_state=0)

x_train.shape

from sklearn.ensemble import RandomForestRegressor
rfr_model = RandomForestRegressor(criterion='mse' , random_state=1 , n_jobs=-1).fit(x_train , y_train)

rfr_pred = rfr_model.predict(x_test)

from sklearn.metrics import mean_squared_error , r2_score

rfr_model.score(x_test , y_test)

# +
rfr_mse = mean_squared_error(rfr_pred , y_test)

rfr_rmse = rfr_mse ** 0.5
rfr_mse
# -

r2_score(rfr_pred , y_test)

#추가검증 포자의 색이 독에 관련이 있을까?
import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x = 'edibility' , hue='spore-print-color' , data=mr)
plt.xticks([0.0 , 1.0] , ['edible' , 'poison'])

sns.countplot(x = 'edibility' , hue='odor' , data=mr)
plt.xticks([0.0 , 1.0] , ['edible' , 'poison'])
# 냄새가 n(nothing)이면 먹지말자


