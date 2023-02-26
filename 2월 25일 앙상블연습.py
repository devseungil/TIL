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
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split , cross_val_score , KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import VotingRegressor

from sklearn.datasets import load_iris

iris = load_iris()

# +
x = iris.data
y = iris.target
x_train , x_test , y_train , y_test = train_test_split(x , y, train_size=0.7 , random_state=123)
sc = StandardScaler().fit(x_train)
x_train_sc = sc.transform(x_train)
x_test_sc = sc.transform(x_test)



# -

dtc = DecisionTreeClassifier()
rfc = RandomForestClassifier()
knn = KNeighborsClassifier()

vc_h = VotingClassifier([('dtc',dtc),('rfc',rfc),('knn',knn)] , n_jobs=-1)
vc_s = VotingClassifier([('dtc',dtc),('rfc',rfc),('knn',knn)] , n_jobs=-1 , voting='soft')
vc_s_h = VotingClassifier([('dtc',dtc),('rfc',rfc),('knn',knn)] , n_jobs=-1 , voting='soft' , weights=[4,3,2])

# +
name = ['dtc','rfc','knn','vc_h','vc_s','vc_s_h']
mod = [dtc,rfc,knn,vc_h,vc_s,vc_s_h]

for n,m in zip(name , mod) :
    m.fit(x_train_sc , y_train)
    pred = m.predict(x_test_sc)
    score = accuracy_score(y_test , pred)
    print(f'{n} , {score}')
# -

print(vc_h.transform(x_test_sc)[3])
print(vc_s.transform(x_test_sc)[3])
print(vc_s_h.transform(x_test_sc)[3])

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
#

from sklearn.ensemble import BaggingClassifier

# +
scores = {}

dtc.fit(x_train_sc , y_train)
result = cross_val_score(dtc , x_test_sc , y_test , cv=KFold(n_splits=5))
score = dtc.score(x_test_sc , y_test)

scores[('dtc' ,'train score')] = result.mean()
scores[('dtc' ,'test score')] = score

# +
bag = BaggingClassifier(dtc , n_estimators=100 , random_state=123).fit(x_train_sc , y_train)
result = cross_val_score(bag , x_test_sc , y_test , cv = KFold(n_splits=5))
score = bag.score(x_test_sc , y_test)

scores[('bag' , 'train score')] = result.mean()
scores[('bag', 'test score')] = score

pd.Series(scores).unstack()
# -

from sklearn.ensemble import StackingClassifier

# +
dtc.fit(x_train_sc , y_train)
dtc_cross_val = cross_val_score(dtc , x_test_sc , y_test , cv=KFold(n_splits=5)).mean()
dtc_score = dtc.score(x_test_sc , y_test)

rfc.fit(x_train_sc , y_train)
rfc_cross_val = cross_val_score(rfc , x_test_sc , y_test , cv=KFold(n_splits=5)).mean()
rfc_score = rfc.score(x_test_sc , y_test)

knn.fit(x_train_sc , y_train)
knn_cross_val = cross_val_score(knn , x_test_sc , y_test , cv=KFold(n_splits=5)).mean()
knn_score = knn.score(x_test_sc , y_test)

stacking = StackingClassifier([('dtc',dtc),('rfc',rfc),('knn',knn)] , final_estimator=RandomForestClassifier())
stacking.fit(x_train_sc , y_train)
stacking_cross_val = cross_val_score(stacking , x_test_sc , y_test , cv=KFold(n_splits=5)).mean()
stacking_score = stacking.score(x_test_sc , y_test)
# -

scores = {}

# +
scores[('dtc' , 'cross_val_score')] = dtc_cross_val
scores[('dtc' , 'acc_score')] = dtc_score
scores[('rfc' , 'cross_val_score')] = rfc_cross_val
scores[('rfc' , 'acc_score')] = rfc_score
scores[('knn' , 'cross_val_score')] = knn_cross_val
scores[('knn' , 'acc_score')] = knn_score
scores[('stacking' , 'cross_val_score')] = stacking_cross_val
scores[('stacking' , 'acc_score')] = stacking_score

pd.Series(scores).unstack()

# -


