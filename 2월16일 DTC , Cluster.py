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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

from sklearn.neighbors import KNeighborsClassifier
# -

# ## DecisionTreeClassifier
# 목표: 단가 높은 다이아몬드 판단하기  
# <br>1. <b>carat_per_price</b> 칼럼을 생성, 캐럿당 가격을 알 수 있는 합성변수를 만들고 
# <br> 상위 25% 값인 약 4950 보다 높으면 1, 나머지는 0 으로하는 target 변수 생성 
# <br>2. cut, color, clarity one-hot인코딩 진행 price 제외한 모든 수치형 변수 독립변수로 추가( 종속변수 제외, 총 27개 변수)
# <br>3. 트레이닝 데이터셋, 테스트 데이터셋 7:3 비율로 생성(random_state=123)
# <br>4. from sklearn.tree import DecisionTreeClassifier 활용 
# <br>(불순도 기준: Gini, Max Depth: 6, Min Samples Splits: 5, Seed: 1234, 그 외: Default)
# <br> 27개 독립변수를 활용하여 target을 분류 예측하는 의사결정 나무 모델 적합 
# <br> 16,182개 샘플이 있는 테스트 셋을 바탕으로 예측하고 실제값과 비교해 f1_score를 구하시오(target 1 기준으로)

dia = pd.read_csv('[통합] 데이터셋//diamonds.csv')

dia

dia['carat_per_price'] = dia['price'] / dia['carat']

dia['target'] = np.where(dia['carat_per_price'] > 4950 , 1 , 0)

dia['target'].value_counts()

dia_dummy = pd.get_dummies(dia[['cut','clarity','color']] , drop_first=True)

df = pd.concat([dia[['target','carat','depth','table','x','y','z']] , dia_dummy] , axis = 1)

df.shape

df_train, df_test = train_test_split(df,  random_state=123, train_size=0.7)

model_tree = DecisionTreeClassifier(max_depth=6, min_samples_split=5, random_state=1234).fit(df_train.drop('target',axis=1),df_train['target'])

pd.DataFrame({'feature' : model_tree.feature_names_in_ , 'imp' : model_tree.feature_importances_}).sort_values('imp',ascending = False)
#분류를하는데 핵심변수 내림차순 / 독립변수넣었을시 종속변수 target이 1이될 확률에 기여가 큰 놈들 내림차순

predict_proba = model_tree.predict_proba(df_test.drop('target',axis = 1))
predict = model_tree.predict(df_test.drop('target' , axis = 1))
pd.DataFrame(predict_proba)

print(classification_report(predict , df_test['target'])) #더미변수 드랍퍼스트 권장 복잡도 줄이기위함

# ## 5. K-Nearest Neighbor
# 전처리
# <br>단계 1: 분석에 사용하지 않을 city, company_size, company_type 컬럼을 제거하시오.
# <br>단계 2: 각 문자형(String Type) 컬럼에 결측치(null/empty space)가 하나라도 존재하는 행(row)은 모두 제거하시오.
# <br>단계 3: experience 컬럼의 값이 ‘>20’ 또는 ‘<1’인 값을 제거하고 experience 컬럼의 타입을 정수형(Integer)으로 변환하시오.
# <br>단계 4: last_new_job 컬럼의 값이 ‘>4’ 또는 ‘never’인 값을 제거하고 last_new_job컬럼의 타입을 정수형(Integer)으로 변환하시오.

ds2 = pd.read_csv('../data/DS_Sample_2.csv')

ds2 = ds2.drop(['city', 'company_size', 'company_type'],axis=1).reset_index(drop=True)

string_list = []
for i in range(ds2.shape[1]) :
    if ds2[ds2.columns[i]].dtype == 'object' :
        string_list.append(ds2.columns[i])

ds2 = ds2.loc[ ds2[string_list].dropna().index.tolist() , : ].reset_index(drop=True)

ds2 = ds2.loc[(ds2['experience'] != '>20') & (ds2['experience'] != '<1') , : ].reset_index(drop=True)
# ds2 = ds2.query("(experience != '>20')&(experience != '<1')").reset_index(drop=True)

ds2['experience'] = ds2['experience'].astype(int)

ds2 = ds2.query("(last_new_job != '>4') & (last_new_job != 'never')")

ds2['last_new_job'] = ds2['last_new_job'].astype(int)

# 문제1 
#
# ‘관련 분야 경험 여부(relevant_experience)’에 따른 ‘이직 희망 여부(target)’를 기술통계량으로 확인하고자 한다.
#
# 관련 분야 경험이 없는(relevant_experience=‘No relevant experience’) 수료자 중 이직을 희망(target=’1’)하는 수료자의 비율을 A, 관련 분야 경험이 있는(relevant_experience=’Has relevant experience’) 수료자 중 이직을 희망(target=’1’)하는 수료자의 비율을 B라 할 때, A/B를 구하시오.
# - 소수점 셋째 자리에서 반올림하여 소수점 둘째 자리까지 기술하시오

pd.crosstab(ds2['relevant_experience'] , ds2['target'])

a = 541/(872+541)
b = 1319/(4790+1319)

round(a/b , 2)

# 문제 2
#
# ‘이직 희망 여부(target)’에 영향을 주는 변수들을 확인하고자 한다. 다음 절차에 따라 로지스틱 회귀분석(Logistic Regression)을 수행하고 질문에 답하시오.
#
# 단계 1: gender, relevant_experience, enrolled_university, education_level, major_discipline 변수로부터 더미 변수들을 생성한다. 단, 각 변수로부터 더미 변수를 생성할 때 마지막으로 등장하는 범주는 제외하도록 한다.
# (여기서 마지막으로 등장하는 범주란, 각 컬럼의 값을 사전 순으로 나열하였을 때 마지막으로 등장하는 값이다. 예를 들어, ‘col’ 변수의 범주가 [‘A’,’C’,’B’,’B’,’C’,’A’]의 값을 가진다면 사전 상의 마지막 값인 C가 제외된다)
#
# 단계 2: 단계 1에서 생성한 더미 변수와 city_development_index, experience, last_new_job, training_hours, target, Xgrp 변수를 결합하여 새로운 데이터셋(데이터셋명: job2_2, 이 데이터셋은 문제 6에서도 활용)을 구성한다. 이 때, target, Xgrp를 제외한 데이터셋의 컬럼은 아래 순서에 따르도록 한다.
# - city_development_index
# - experience
# - last_new_job
# - training_hours
# - gender의 더미 변수
# - relevant_experience의 더미 변수
# - enrolled_university의 더미 변수
# - education_level의 더미 변수
# - major_discipline의 더미 변수
#
# 단계 3. 단계 2에서 구성한 데이터셋 job2_2로 다음 조건에 따라 상수항(Intercept)이 포함된 로지스틱 회귀분석을 수행한다.
#
# - 종속 변수 : target
# - 독립 변수(총 16개) : target과 Xgrp를 제외한 나머지 변수
# - 회귀식에 포함되는 독립 변수의 순서를 컬럼의 순서와 일치시킨다.
# - C=100000, random_state=1234
# - 상수항을 제외한 나머지 변수들에 대한 Odds Ratio중 가장 큰 값을 기술하시오.

ds2_1 = pd.get_dummies( ds2[["gender", "relevant_experience", "enrolled_university", 
               "education_level", "major_discipline"]])

ds2_1 = ds2_1.drop( ['gender_Other', 'relevant_experience_No relevant experience','enrolled_university_no_enrollment',
           'education_level_Phd','major_discipline_STEM'], axis=1)

ds2_2 = pd.concat([ds2[["city_development_index", "experience", "last_new_job", "training_hours", 
            "target", "Xgrp" ]], ds2_1], axis=1)

model2_2 = LogisticRegression(C=100000, random_state=1234, fit_intercept=True)

model2_2.fit(ds2_2.drop(['target', 'Xgrp'],axis=1), ds2_2['target'])

np.exp(model2_2.coef_)[0]

pd.DataFrame({'oddratio':np.exp(model2_2.coef_)[0]}).sort_values('oddratio', ascending=False)

# (job2_2 를 이용하여 ) 전체 데이터를 T rain 과 Test S et 으로 나누고 , Train S et 으로 학습한 모델을 Test S et 에
# 적용하여 모델을 평가하고자 한다 다음 절차에 따라 분석을 수행하고 질문에 답하시오
#
# 단계
# 1 : 문제 5 번 2 단계에서 구성한 데이터셋 job2_2 에서 Xgrp 컬럼의 값이 'train‘ 인 경우 T rain S et 으로 ‘test' 인
# 경우 Test S et 으로 정의하여 분할한다
#
# 단계
# 2: 아래 가이드에 따라 T rain S et 으로 K NN 분류 모델을 학습하고 이 모델을 Test S et 에 적용한다
# - 종속 변수 이직 희망 여부 target)
# - 독립 변수 (총 16 개) 이직 희망 여부 (target) 와 Train/Test set 구분 변수 (Xgrp) 를 제외한 모든 변수
# - Euclidean 거리 기준 가장 가까운 5 개 데이터의 이직 희망 여부 (target) 를 활용하여 예측

ds2_3 = ds2_2.copy()

ds2_3_train = ds2_3.query("Xgrp == 'train'")
ds2_3_test = ds2_3.query("Xgrp == 'test'")
print(ds2_3_train.shape)
print(ds2_3_test.shape)

from sklearn.neighbors import KNeighborsClassifier

model2_3 = KNeighborsClassifier( n_neighbors=5)

model2_3.fit(ds2_3_train.drop(['target' , 'Xgrp'],axis=1) , ds2_3_train['target'])

predic2_3 = model2_3.predict(ds2_3_test.drop(['target', 'Xgrp'],axis=1))

pd.crosstab( predic2_3 , ds2_3_test['target'], margins=1)

round((1897+108)/2816 , 2) #정확도

# ## 1-1. Clustering - Hierarchical Clustering 계층적 군집화

# 전처리 과정

df_h = pd.read_csv('../data/hk_221206.csv')

df_h = df_h[['gender', "age", "company", "grades", "salary", "expenditure" ]]

st_h = StandardScaler().fit( df_h[['age', 'salary', 'expenditure']])
st_h_table = pd.DataFrame(st_h.transform(df_h[['age', 'salary', 'expenditure']]), columns=['age_st', 'salary_st', 'expenditure_st'])
st_h_table

df_h_dummy = pd.get_dummies( df_h[['gender', 'company', 'grades']])

basetable_h = pd.concat( [df_h, st_h_table, df_h_dummy], axis=1)
basetable_h

from sklearn.cluster import AgglomerativeClustering

basetable_h_cluster_1 = basetable_h.iloc[ : , 6:]

cluster_h = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward').fit(basetable_h_cluster_1)

cluster_h.labels_

cluster_h.fit_predict(basetable_h_cluster_1)

# 13개 변수로 만들어진 클러스터 모델에 독립변수들 넣어서 어느군집에 해당하는지 예측
basetable_h['cluster_hier'] = cluster_h.fit_predict( basetable_h_cluster_1)

basetable_h['cluster_hier'].value_counts()

#cluster 별 나이, 연봉, 지출 평균
basetable_h.groupby(['cluster_hier'])[['age', 'salary', 'expenditure']].mean()







# ## 1-6. Clustering 예측 ( Kmeans )
# 150개 데이터 셋을 바탕으로 군집분석을 실시 하였다.
#
# n_clusters=3, random_state=123
#
# <br> 150개 외 추가 데이터 셋 샘플을 추가 할 경우, 모델을 바탕으로 기존 군집분석을 바탕으로 Cluster를 분류 할 수 있다. 
# <br> 모델은 Kmeans 알고리즘을 통해 3개 cluster로 분류한 cluster_1_2 모델을 활용한다.
#
# <br> 데이터셋 샘플 - 성별:남성 / age:33 / company :C / grades: B / salary : 4500 / expenditure: 2975
#
# <br> <b>순서도</b>
# <br> 1.수치형 변수 표준화 -> 2. 더미변수 확인 -> 3. 데이터 프레임에 맞춰 데이터 셋 준비 -> 4.Cluster 예측 
#
# <br> 군집분석시 활용할 변수는 <b>gender, age, company, grades, salary, expenditure</b> 이다.   
# <br> 이때 수치형 변수 age, salary, expenditure는 정규화를 진행하고 정규화한 칼럼은 각각 age_st, salary_st, expenditure_st로 명명한다
# <br> 명목형 변수 gender, company, grades는 더미변수화 한다.(drop_first 옵션 false, 순서는 표기된 대로 진행할 것) 
# <br> 전체 데이터셋 순서는 표준화한 age, salary, expenditure와 나머지 gender, company, grades 더미변수다.  
#
# <br>
# <br> 위 전처리를 마친 후 데이터셋 이름은 <b>basetable1</b>로 명명한다

df = pd.read_csv('[통합] 데이터셋/hk_221206.csv')
df = df[['gender', "age", "company", "grades", "salary", "expenditure"]]

st = StandardScaler().fit(df[['age','salary','expenditure']])
st_table = pd.DataFrame(st.transform(df[['age','salary','expenditure']]), columns=['age_st','salary_st','expenditure_st'])
st_table

dummy_table = pd.get_dummies(df[['gender','company','grades']])

basetable1 = pd.concat([df , st_table, dummy_table], axis = 1)

base_cluster = basetable1.iloc[ : , 6: ]
base_cluster

cluster_1 = KMeans(n_clusters=3, random_state=123).fit(base_cluster)
cluster_1.labels_

basetable1['kmean_cluster'] = cluster_1.labels_

# +
# basetable1['kmean_cluster'] = basetable1['kmean_cluster'].map({'A' : 0, 'B' : 1 , 'C' : 2})
# -

basetable1

성별:남성 / age:33 / company :C / grades: B / salary : 4500 / expenditure: 2975

sample = pd.DataFrame({'age' : [33], 'salary' : [4500] , 'expenditure' : [2975]})

sample_st = pd.DataFrame(st.transform(sample), columns=['age_st','salary_st','expenditure_st'])

sample_st

sample_dummy = pd.DataFrame([[0,0,0,0,0,0,0,0,0,0]] , columns= dummy_table.columns)

sample_dummy['gender_M'] = 1
sample_dummy['company_C'] = 1
sample_dummy['grades_B'] = 1
sample_dummy

sample_cluster = pd.concat([sample_st , sample_dummy] , axis = 1)

sample_cluster

print("샘플은",cluster_1.predict(sample_cluster)[0],"번 군집에 속한다")

cluster_1.predict(sample_cluster)
