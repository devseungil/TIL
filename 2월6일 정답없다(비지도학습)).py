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

# 정답없는경우 데이터로드 - > 전처리 ->  EDA -> 차원축소(PCA) -> 클러스터링 수 결정 ->모델학습 -> 평가
#
# 예측 정밀도를 평가할수없음 타겟없이 데이터만으로

from sklearn.datasets import load_wine
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

wine = load_wine()

df_wine = pd.DataFrame(data=wine.data , columns=wine.feature_names)
df_wine.info()

train , test = train_test_split(df_wine , train_size=0.7 , random_state=0)

train.describe()
#alcohol malic_acid 평균값의 스케일이다름 -> 최소최대값도 차이가 큼 스케일링 해줘야함
# 이상치가 없다면 정규화하기
# 이상치 확인 -> pairplot

import seaborn as sns
# sns.pairplot(train)
# y값 폭이 크다? => 이상치(돌출값)이 있따 현재는업슴

from sklearn.preprocessing import MinMaxScaler

ms = MinMaxScaler().fit(train)

train_ms = pd.DataFrame(ms.transform(train) , columns=train.columns)

sns.pairplot(train_ms)

# 차원축소(PCA) : 특징량(X) -> 줄이기 
#
# 시각화가 쉬워짐 , 클러스터링이 쉽다

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca.fit(train_ms)

train_2d = pd.DataFrame(pca.transform(train_ms) , columns=['pca_1' , 'pca_2'])
train_2d[:3]

sns.scatterplot(x='pca_1' , y='pca_2' , data = train_2d)

from sklearn.cluster import KMeans

# +
import matplotlib.pyplot as plt
lst_sse = []

for i in range(1,11) :
    km = KMeans(n_clusters=i , random_state=0)
    km.fit(train_2d)
    lst_sse.append(km.inertia_) # 클러스터 內 분산, 값이 적을수록 좋다
    
plt.plot(range(1,11) ,lst_sse, marker = 'o')
plt.xlabel('number of cluster')
plt.ylabel('sse')
plt.show()
# -

km = KMeans(n_clusters=3 , random_state=0).fit(train_2d)

pred_km = km.predict(train_2d)

train_2d['label'] = pred_km

train_2d

sns.scatterplot(x='pca_1' , y='pca_2' , hue='label' ,data=train_2d , palette='Set1')

train['label'] = pred_km

sns.pairplot(train , hue='label',palette='Set1')




