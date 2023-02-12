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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

base1 = pd.read_csv('../data/hk_221206.csv')
df = base1.copy()

# %pwd

pd.crosstab(base1['gender'], base1['company'], normalize=0) #젠더기준 컴패니에 얼마씩있는지

pd.crosstab(base1['gender'], base1['company'], normalize=1) #컴패니기준 젠더의 비율

pd.crosstab(base1['gender'], base1['company'], normalize=True) #전체기준 

# ## 알파 퀴즈)
# 전체 소득 대비 소비액 비율을 나타내는 합성 변수를 만들고자 한다.
#
# 수식 : expenditure_per_salary = expenditure / salary
#
# expenditure_per_salary 합성변수를 만들고 해당 변수의 company 그룹별 평균을 구하시오

df['expenditure_per_salary'] = df['expenditure']/df['salary']

df.groupby('company')['expenditure_per_salary'].mean().reset_index()

# ## 1-1. 단일 회귀 statemodels - ols()
#
# 연봉으로 지출액을 예측할 수 있을까 ?

from statsmodels.formula.api import ols

model1_1 = ols(data= df, formula=('expenditure ~ salary')).fit()

# 모델 summary 결정계수 / 회귀 계수 등 확인 
model1_1.summary() 

# ## 결정계수 : 0.945 ( 0.5는 넘어야 신뢰성o ) 1이 가장완벽
#
# ## 검정통계량 : F-statistic:	4273 / pvalue = 2.40e-158 (0.05보다작다 = 통계적으로 유의미하다)
#
# ## Durbin-Watson:	1.840 잔차의독립성 1.5~2.5 = ok
#
# ## 회귀식 : 0.9781 x (연봉) - 1246.9920

model1_1.params


def linear1_1(x):
    return (model1_1.params[1] * x) + model1_1.params[0] 
linear1_1(4100)

predict1_1 = model1_1.predict(df['salary'])
predict1_1[:5] #연봉에따른 지출예측치 인덱스0부터 촤르륵

# ## 1-2. 단일 회귀 sklearn.linear_model
#
# 연봉으로 지출액을 예측할 수 있을까 ?

from sklearn.linear_model import LinearRegression

#사이킷런은 독립변수 종속변수 순임
model1_2 = LinearRegression().fit(df[['salary']], df['expenditure'])
model1_2

print(model1_2.coef_)
print(model1_2.intercept_)

# statemodels vs. sklearn 차이 
# - statemodels는 summary 표 활용 가능(친절함) 
# - 입력값의 차이( statemodels ols의 경우 formula 문법이 있음 / sklearn는 fit() 활용) 

# ## 1-3. 단일 회귀 train_test_split / statemodels - ols()
#
# train, test 로분할
#
# 연봉으로 지출액을 예측할 수 있을까 ?

# +
from sklearn.model_selection import train_test_split

df_train1, df_test1 = train_test_split(df, train_size= 0.7, random_state=123)
# -

model1_3 = ols(data= df_train1, formula=('expenditure ~ salary')).fit()

model1_3.summary()

model1_3.params

predict1_3 = model1_3.predict(df_test1['salary'])
predict1_3.reset_index()[:5]

# ## Quiz 단일회귀 train_test_split / statemodels - ols() 
#
# train, test data 분할
#
# 나이로 연봉을 예측할 수 있을까 ? 



# ## 1-5. 단일회귀 train_test_split /  sklearn.linear_model
#
# train, test data 분할
# <br> 나이로 연봉을 예측할 수 있을까 ?

# +
model1_5 = LinearRegression().fit(df_train1[['age']], df_train1['salary'])

predict1_5 = model1_5.predict(df_test1[['age']])

predict1_5[:5]
# -

# ## 2-1. 다중회귀 statemodels -ols()
# ## (추가) 다중 회귀 가정 4가지 선형성, (정규성, 등분산, 독립성) = 오차(잔차)
#
#
# train, test data 분할
# <br> 종속변수 : 지출액 / 독립변수 : 연봉과 나이

model2_1= ols(data= df_train1, formula=('expenditure ~ age + salary')).fit()

model2_1.summary()

predict2_1 = model2_1.predict(df_test1[['age','salary']])
predict2_1.reset_index()[:3]

# ## 2-2 sklearn.linear_model 
#
# train, test data 분할
# <br> 연봉과 지출액으로 나이를 예측할 수 있을까 ?

model2_2 = LinearRegression().fit(df_train1[['salary', 'expenditure']], df_train1['age'])
predict2_2 = model2_2.predict(df_test1[['salary','expenditure']])
predict2_2[:5]

print(model2_2.intercept_)
print(model2_2.coef_)

# 변수별 회귀계수 확인하는 데이터프레임 제작 
pd.DataFrame({'feature':model2_2.feature_names_in_, 'coef':model2_2.coef_}).sort_values('coef', ascending=False)

# ## 2-3 선형회귀 모형 명목형 변수의 처리
# ## 명목형 변수(문자형)를 수치형변수로 변환
# get_dummy

# 해당 column만
pd.get_dummies(data = df[['company', 'grades']]) # 사전 순서로 정렬

# 전체 column + 해당 column, 원본 column은 삭제됨
df_dummy = pd.get_dummies(data = df, columns=['company', 'grades'])
df_dummy

# ## 2-4 다중회귀 sklearn.linear_model 
#
# 회사와(company, 범주형) 연봉(salary, 연속형)으로 지출액을 예측할 수 있을까 ?

model2_4 = LinearRegression().fit(df_dummy[['company_A', 'company_B', 'company_C', 'salary']], df_dummy['expenditure'])
predict2_4 = model2_4.predict(df_dummy[['company_A', 'company_B', 'company_C', 'salary']])
predict2_4[:5]

pd.DataFrame({'feature':model2_4.feature_names_in_, 'coef': model2_4.coef_.round(2)})

model2_4.intercept_

# ## 선형회귀 모델의 평가 수가 낮을수록 좋음
#
# MAE, MSE, RMSE

model_age_to_expd = LinearRegression().fit(df_train1[['age']], df_train1['expenditure'])
predict_age_to_expd = model_age_to_expd.predict(df_test1[['age']])

from sklearn.metrics import mean_absolute_error, mean_squared_error

mean_absolute_error(df_test1['expenditure'], predict_age_to_expd)

mean_squared_error(df_test1['expenditure'], predict_age_to_expd)

mean_squared_error(df_test1['expenditure'], predict_age_to_expd) ** 0.5

# ## 3-1 표준화, 정규화 
# 표준화 : min-max 단위를 고르게 하기 위하여 모든 값을 0~1사이로 바꾸는 것 (최소값0 최대값1)
# <br> sklearn.preprocessing / MinMaxScaler 활용

from sklearn.preprocessing import MinMaxScaler

df_n = df[['height', 'age', 'salary', 'expenditure']]
df_n

sc_minmax = MinMaxScaler()

df_minmax = pd.DataFrame( sc_minmax.fit_transform(df_n), columns = df_n.columns) 
df_minmax

# ### 정규화
# 정규화: StandardScaler 모든 변수의 값을 평균이 0이고 분산이 1인 정규 분포로 변환
# <br> sklearn.preprocessing / MinMaxScaler 활용

from sklearn.preprocessing import StandardScaler

sc_stan = StandardScaler()
df_stan = pd.DataFrame( sc_stan.fit_transform(df_n), columns = df_n.columns) 
df_stan

# ## 4-2 Feature Engeering
#
# 보스톤 집값 데이터 셋(출처: sklearn 라이브러리) 

boston = pd.read_csv('[통합] 데이터셋\\sklearn_boston.csv')
boston.info()

" + ".join( boston.columns.drop('price'))

df_train2, df_test2 = train_test_split( boston, train_size=0.8, random_state=123)

model4_2 = ols(formula=('price ~ crim + zn + indus + chas + nox + rm + age + dis + rad + tax + ptratio + b + lstat'), data=df_train2).fit()

model4_2.summary() # p-value 값 0.05 미만 / coef 절대값 0.5이상  = 통계상 의미가있다

# ## quiz 2) 다이아몬드 데이터 셋 활용
#
# - diamond.csv 파일을 읽고 (df_dia)
# train, test로 분리하시오 (random_state=123) (df_dia_train, df_dia_test)
#
# - df_dia_train으로 선형회귀분석을 실시하고 아래에 답하시오
#
#   2_1. 종속변수(price), 독립변수(carat, depth)일때 독립변수의 회귀계수를 구하시오
#
#   2_2. 종속변수(price), 독립변수(carat, depth, color)일때 df_dia_test price의 예측값 평균을 구하시오
#
#     'color'는 더미변수로 변형하고 가변수를 생성시 마지막 변수 하나(drop first = ture)를 제거하시오 
#
#   2_3. 2_2 조건으로 df_dia_test의 값이 (carat: 1, color: 'E' , depth: 50)일때 price 예측값을 구하시오 

# 파일 로딩 
df_dia = pd.read_csv('../data/diamonds.csv')

df_dia[:2]

dia_train , dia_test = train_test_split(df_dia , train_size=0.7 , random_state=123)
dia_train['color'].unique()

# dia_model = ols(formula=('price ~ carat + depth'), data=dia_train).fit()
# dia_model.summary() 7770.1651 , -106.544
dia_model = LinearRegression().fit(dia_train[['carat','depth']] , dia_train['price'])
dia_model.coef_

dia_model2 = ols(formula=('price ~ carat + depth + color'), data=dia_train).fit()
# dia_model2.predict(dia_test[['carat','depth','color']]).mean()

df_dummy = pd.get_dummies(data=df_dia , columns=['color'])
df_dummy = df_dummy.drop(['cut' , 'clarity' , 'table' , 'x','y','z'] , axis = 1)

dia_train2 , dia_test2 = train_test_split(df_dummy , train_size=0.7 , random_state=123)
dia_train2

dia_model3 = LinearRegression().fit(dia_train2.drop('price' , axis = 1) , dia_train2['price'])

# +
# 2_3. 독립변수 직접입력 방식

dia_model3.predict([[1,50,0,1,0,0,0,0,0]])
