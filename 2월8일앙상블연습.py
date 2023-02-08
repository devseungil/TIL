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

from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression #lr
from sklearn.ensemble import RandomForestClassifier #rf 
from sklearn.neighbors import KNeighborsClassifier  #knn
from sklearn.ensemble import VotingClassifier
import warnings
warnings.filterwarnings('ignore')

iris = load_iris()

x = iris.data
y = iris.target
sc = StandardScaler().fit(x)
x_sc = sc.transform(x)
y

x_train , x_test , y_train , y_test = train_test_split(x_sc, y , train_size=0.7 , random_state=123)

lr = LogisticRegression()
rfc = RandomForestClassifier()
knn = KNeighborsClassifier()
vc1 = VotingClassifier(estimators=[('lr',lr),('rfc',rfc),('knn',knn)])
vc2 = VotingClassifier(estimators=[('lr',lr),('rfc',rfc),('knn',knn)] , voting='soft')
vc3 = VotingClassifier(estimators=[('lr',lr),('rfc',rfc),('knn',knn)] , voting='soft' , weights=[5,4,3])

# +
ests = [lr,rfc,knn,vc1,vc2,vc3]
names = ['lr','rfc','knn','hard','soft','soft and weight']

for est,name in zip(ests,names) :
    est.fit(x_train,y_train)
    print(name , est.score(x_test , y_test))
# -

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

can = load_breast_cancer(as_frame=True)

x = can.data
y = can.target

sc = StandardScaler().fit(x)

x_sc = sc.transform(x)

kfold = KFold(n_splits=5)

x_train , x_test , y_train , y_test = train_test_split(x,y, train_size=0.7 , random_state=123)

dt = DecisionTreeClassifier().fit(x_train , y_train)

print("test score :",dt.score(x_test , y_test))

print("train score :",(cross_val_score(dt , x_test , y_test , cv=kfold)).mean())

bag = BaggingClassifier(dt , n_estimators=100 , n_jobs=-1).fit(x_train , y_train)

bag.score(x_test , y_test)

result = cross_val_score(bag , x_test , y_test , cv=kfold)
result.mean()

# +

from sklearn.svm import SVC
from sklearn.ensemble import StackingClassifier
# -

svc = SVC(probability=True).fit(x_train , y_train)
print("test score :",svc.score(x_test,y_test))
print("train score :", (cross_val_score(svc , x_test , y_test , cv=kfold)).mean())

log = LogisticRegression().fit(x_train , y_train)

print("test score:",log.score(x_test , y_test))
print("train score:",(cross_val_score(log , x_test , y_test , cv=kfold)).mean())

stack = StackingClassifier(estimators=[('svc',svc),('log',log),('dt',dt)] , final_estimator=LogisticRegression()).fit(x_train , y_train)

stack.score(x_test , y_test)

(cross_val_score(stack , x_test , y_test  , cv=kfold)).mean()

score = {}
score[('dt') , ('test score')] = dt.score(x_test , y_test)
score[('dt') , ('train score')] = (cross_val_score(dt , x_test , y_test , cv=kfold)).mean()
score[('svc') , ('test score')] = svc.score(x_test , y_test)
score[('svc') , ('train score')] = (cross_val_score(svc , x_test , y_test , cv=kfold)).mean()
score[('log') , ('test score')] = log.score(x_test , y_test)
score[('log') , ('train score')] = (cross_val_score(log , x_test , y_test , cv=kfold)).mean()
score[('stack') , ('test score')] = stack.score(x_test , y_test)
score[('stack') , ('train score')] = (cross_val_score(stack , x_test , y_test , cv=kfold)).mean()

pd.Series(score).unstack()


