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

# +
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

hk = pd.read_csv('[통합] 데이터셋\\hk_221206.csv')
hkna = pd.read_csv('[통합] 데이터셋\\hk_221206_na.csv')
# -

from scipy.stats import chi2_contingency

# # 카이제곱검정의 귀무가설은 서로 독립이다
# (크로스테이블 범주형데이터 구성비 )

cross1 = pd.crosstab(hk['gender'],hk['company'])
cross1 #자유도 = company 의 종류 -1 x gender의 종류 -1  = 2x1 = 2

chi2_contingency(cross1) #2번째가 pvalue 0.05보다 큼 귀무가설채택 = 컴패니와 젠더는 독립이다

cross2 = pd.crosstab(hk['gender'],hk['grades'])
chi2_contingency(cross2)

hkna.isna().sum()

hkna.fillna(100).mean()

hkna.dropna().mean()

q1, q3 = hk['expenditure'].quantile([0.25,0.75])
IQR = q3 - q1
low = q1- IQR * 1.5
high = q3 + IQR * 1.5
low, high

hk['expenditure'] > high

from sklearn.model_selection import train_test_split
train, test = train_test_split(hk, train_size = 0.7, random_state = 123) #랜덤샘플링 랜덤스테이트하면 고정으로뽑힘

train[:5]

hkin = hk.reset_index()
hkin[hkin['index'] % 5 == 0][:5]

sns.histplot(hk[['height','age']])

from sklearn.preprocessing import MinMaxScaler # 최대 최소 변환
from sklearn.preprocessing import StandardScaler #(Z-score 변환)


# 표준화(Minmax) 혹은 정규화(Standar...)로 변수들의 단위를 맞춰주기 위함입니다
#

# +
minmax = MinMaxScaler().fit_transform(hk[['height','age']])
standard = StandardScaler().fit_transform(hk[['height','age']])

AAmodel = MinMaxScaler().fit(X = df_train[['x1', 'x2']])
AAmodel_train = AAmodel.transform(X = df_train[['x1', 'x2']])



hk[['height_s', 'age_s']] = standard
# -

hk

sam = pd.read_csv('[통합] 데이터셋\\DS_Sample_3.csv')

import datetime
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from sklearn.linear_model import LinearRegression

samyear= sam['AirDate'].astype('datetime64').dt.year

sam['group'] = np.where(samyear.between(2005,2007), 'A' , np.where(samyear.between(2008,2010), 'B', 'C'))
sam['group2'] = pd.cut(samyear, bins = [2004,2007,2010,2013], labels=['a','b','c'])

sam['success'] = sam['Rating'] * sam['Num_Votes']
int(max(sam.groupby('DirectedBy')['success'].mean()))

sam

model = ols(formula='Num_Votes ~ group', data=sam).fit()
anova_lm(model) #검정통계값 + 자유도 (F + df)


