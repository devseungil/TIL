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
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

wine = load_wine()

df_wine = pd.DataFrame(data = wine.data , columns=wine.feature_names)
df_wine

df_train , df_test = train_test_split(df_wine , train_size=0.7 , random_state=123)

df_train.shape

ms = MinMaxScaler().fit(df_train)
train_ms = pd.DataFrame(ms.transform(df_train) , columns=df_train.columns)

train_ms

pca = PCA(n_components=2).fit(train_ms)
train_2d = pd.DataFrame(pca.transform(train_ms),columns=['pca_1' , 'pca_2'])


train_2d

for i in range(2,8) :
    km = KMeans(n_clusters=i , random_state=123).fit(train_2d)
    print(i , silhouette_score(train_2d , km.labels_))


km = KMeans(n_clusters=3 , random_state=123).fit(train_2d)
train_2d['target'] = km.labels_

train_2d

import seaborn as sns

sns.scatterplot(x='pca_1' , y='pca_2' , hue='target' ,data=train_2d)


