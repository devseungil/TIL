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

import warnings
warnings.filterwarnings('ignore')
# -

hk = pd.read_csv('../data/\\hk_221206.csv')
iris = pd.read_csv('../data/\\iris.csv')
bike = pd.read_csv('../data/\\bike.csv')

from scipy.stats import ttest_1samp
from scipy.stats import ttest_ind

hk['age'].mean()

ttest_1samp(hk['age'], popmean=38) # 모평균을 38로 봤을때 pvalue가 0.05보다 작으므로 모평균과 표본의평균은 같지않다 귀무가설 기각

hk.groupby(['company'])['salary'].mean()

AS = hk[hk['company'] == 'A']['salary']
BS = hk[hk['company'] == 'B']['salary']
ttest_ind(AS,BS) #두 표본의 평균은 같지않다 귀무가설 기각

# 1) iris 데이터를 사용하여('iris.csv') species column 'virginica'의 'sepal_width' 모평균이 3.14와 같은지 가설을 수립하고 유의수준 0.05에서 검정하시오
#
# 2) 'setosa'와 'versicolor'의 sepal_length 평균이 같은지 가설을 수립하고 유의수준 0.05에서 검정하시오
#

vir = iris[iris['species'] == 'virginica']['sepal_width']
ttest_1samp(vir , popmean=3.14) #모평균과 같지않음 귀무가설 기각

iris['species'].unique()

setosa = iris[iris['species'] == 'setosa']['sepal_length']
versicolor = iris[iris['species'] == 'versicolor']['sepal_length']
ttest_ind(setosa, versicolor) #평균이 같지 않음 귀무가설 기각

from scipy.stats import f_oneway
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

A = hk[hk['company'] == 'A']['salary']
B = hk[hk['company'] == 'B']['salary']
C = hk[hk['company'] == 'C']['salary']
f_oneway(A,B,C) #3그룹의 평균은 같지 않다. pvalue < 0.05

Aa = hk[hk['company'] == 'A']['age']
Ba = hk[hk['company'] == 'B']['age']
Ca = hk[hk['company'] == 'C']['age']
f_oneway(Aa,Ba,Ca)

model1 = ols(formula='age ~ company', data=hk).fit()
anova_lm(model1) # PR이 0.05보다 작으므로 모든 종류의 컴패니별 나이의 평균은 서로 같지않다 귀무가설 기각

# 그런데 여기서 만약 p-value가 0.05보다 작다면? 즉, 집단들의 평균이 모두 똑같지 않다는 뜻인데 이 때는 사후 검정을 통해 대체 어떤 집단이 차이가 있는지 확인해야 한다.

from statsmodels.stats.multicomp import pairwise_tukeyhsd

posthoc = pairwise_tukeyhsd(hk['age'], hk['company'])
print(posthoc)
#reject 가 트루면 귀무가설 기각하라는 뜻 (평균이 같지않다) 즉 이표에선 컴패니 A B C 서로 전부 평균이 달라서 기각했단뜻

# +
a_grade = hk[hk['grades'] == 'A'].salary
b_grade = hk[hk['grades'] == 'B'].salary
c_grade = hk[hk['grades'] == 'C'].salary
d_grade = hk[hk['grades'] == 'D'].salary
f_grade = hk[hk['grades'] == 'F'].salary

f_oneway(a_grade,b_grade,c_grade,d_grade,f_grade)
# pvalue 0.05 보다 큼 귀무가설채택 = 5개의 등급별 연봉의 평균이 같다.
# -

post = pairwise_tukeyhsd(hk['salary'], hk['grades'])
print(post)

# # Quiz 2
#
# 1) 'setosa' , 'versicolor', 'virginica'의 sepal_length 평균이 같은지 가설을 수립하고 유의수준 0.05에서 검정하시오
#
#
# 2) bike 데이터(bike.cvs)를 사용하여, 요일별 registered 평균이 같은지 가설을 수립하고 유의수준 0.05에서 검정하시오
#
# 3) 평균이 같지 않을때, 평균이 유의수준 0.05에서 차이나지 않는 조합(False)은 몇 개인가 ? 

iris['species'].unique()

s = iris[iris['species'] == 'setosa']['sepal_length']
ver = iris[iris['species'] == 'versicolor']['sepal_length']
vir = iris[iris['species'] == 'virginica']['sepal_length']
f_oneway(s, ver, vir) #귀무가설 기각

model4 = ols(formula='sepal_length ~ species', data=iris).fit()
anova_lm(model4)

bike

bike['day'] = pd.to_datetime(bike['datetime']).dt.day_name()
# bike.groupby('day')['registered'].mean()
model2 = ols(formula= 'registered ~ day', data=bike).fit()
anova_lm(model2) #귀무가설 기각

po = pairwise_tukeyhsd(bike['registered'], bike['day'])
print(po)

# 1) iris 데이터를 사용하여('iris.csv') species column 'virginica'의 'sepal_width' 모평균이 3.14와 같은지 가설을 수립하고 유의수준 0.05에서 검정하시오
#
# 2) 'setosa'와 'versicolor'의 sepal_length 평균이 같은지 가설을 수립하고 유의수준 0.05에서 검정하시오

vir_width = iris[iris['species']=='virginica']['sepal_width']
ttest_1samp(vir_width , popmean=3.14 # pv < 0.05 귀무가설 기각 모집단평균과 같지 않다

seto_width = iris[iris['species'] == 'setosa']['sepal_width']
ver_width = iris[iris['species'] == 'versicolor']['sepal_width']
ttest_ind(seto_width,ver_width ) # pv < 0.05 귀무가설 기각 두개의 평균이 같지 않다고 검정

# 1) 'setosa' , 'versicolor', 'virginica'의 sepal_length 평균이 같은지 가설을 수립하고 유의수준 0.05에서 검정하시오
#
#
# 2) bike 데이터(bike.cvs)를 사용하여, 요일별 registered 평균이 같은지 가설을 수립하고 유의수준 0.05에서 검정하시오
#
# 3) 평균이 같지 않을때, 평균이 유의수준 0.05에서 차이나지 않는 조합(False)은 몇 개인가 ? 

mo = ols(formula='sepal_width ~ species' , data=iris).fit()
anova_lm(mo) # pv < 0.05 귀무가설 기각 세개의 평균이 같지 않다.

f_oneway(vir_width, seto_width, ver_width )

bike['date'] = pd.to_datetime(bike['datetime']).dt.day_name()
mmm = ols(formula='registered ~ date', data=bike).fit()
anova_lm(mmm)

hi = pairwise_tukeyhsd(bike['registered'], bike['date'])
print(hi)

# ## Quiz
#
# bike 데이터(bike.cvs)를 사용하여
#
# 1) temp, atemp, humidity, registered의 상관 계수중 가장 높은것은 ?
#
# 2) season별로 atemp와 자전거 대여 숫자(casual)와의 상관분석을 실시하고 상관 계수가 가장 높은 계절을 구하시오
#
# 3) 날씨가 맑은날(weather = 1) 과 그렇지 않은날 온도(temp)와 자전거 대여 숫자(casual)의 상관계수의 절대값은 얼마인가 ?

bike[['temp','atemp','humidity','registered']].corr().round(2)

bike[['season','atemp','casual']].groupby('season').corr().round(2).reset_index().sort_values('atemp', ascending = False)
bike[['season','atemp','casual']].groupby('season').corr().round(2).reset_index().nlargest(10, 'atemp')

(bike['weather'] == 1)+0 # True False 가 숫자로 변함 1 과 0 으로
bike['weather_1'] = (bike['weather'] == 1)+0
bike[['temp', 'casual', 'weather_1']].groupby('weather_1').corr().reset_index()


