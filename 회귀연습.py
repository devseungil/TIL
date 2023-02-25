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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report

# ## $\alpha$(알파) 퀴즈: 
#
# <br>기본 데이터셋 df에 차량 소유 데이터셋을 join한다.( 차량 소유 데이터 파일은 hk_221206_car.csv 이며 left 조인) 
# <br>차량 정보는 배기량에 의해 A/B/C/D/E/F 타입으로 되어 있으며 차량이 없는 경우 none으로 되어 있다. 
# <br>6개 차량 타입을 A, B 인 경우 SS로 변경, C, D 인 경우 MM로 변경, E, F인 경우 LL로 변경한다.(컬럼명은 car_type 동일)
# <br>car_type이 none인 경우 결측치로 판단하여 제외한다. 
# <br>전처리 후 데이터 셋 명칭을 basetable1으로 명명한다. 
# <br>
# <br><b>문제: car_type별 연봉 평균을 각각 구하시오(SS/MM/LL 3가지 경우 확인) </b>

car = pd.read_csv('data/hk_221206_car.csv')
df = pd.read_csv('data/hk_221206.csv')

car['car_type'] = car['car_type'].map({'A' : 'SS' , 'B' : 'SS' , 'C' : 'MM' , 'D' : 'MM' , 'E' : 'LL' , 'F' : 'LL'})

basetable = pd.merge(df , car , how='left' , on='name').dropna().reset_index(drop=True)

basetable.groupby('car_type')['salary'].mean()

# ## 1-0. 로지스틱 회귀분석 워밍업 - 종속변수: 성별 
# basetable1 236개 샘플을 활용하여 종속변수를 gender로 독립변수 height, age, salary, expenditure 3개로 
# <br> 구성하는 로지스틱 회귀분석을 설계 
# <br>
# <br>from sklearn.linear_model import LogisticRegression 활용 
# <br>(파라미터 가이드: Seed=1234, Solver='newton-cg', 나머지: Default)
# <br>

ex1 = basetable[['height','age','salary','expenditure']]
ex2 = basetable['gender']

model1 = LogisticRegression(solver='newton-cg' , random_state=1234).fit(ex1 , ex2)

pd.DataFrame({'feature' : model1.feature_names_in_ , 'coef' : model1.coef_[0].round(3)})

# ## 1-1. 로지스틱 회귀분석 sklearn 라이브러리 활용
# basetable1 236 rows × 11 columns
# <br>종속변수 : car_type( LL 여부 yes:1, no:0) 
# <br>독립변수 : salary, expenditure, company(dummy변수)  
#
# 독립변수의 수치가 ~~ 일때 자동차타입이 LL일 확률
#
# sklearn 라이브러리 활용, 파라미터 값 C=100000 ,solver='newton-cg' 적용

ex1_y = np.where(basetable['car_type'] == 'LL' , 1 , 0)
ex1_x = pd.get_dummies(basetable[['salary' , 'expenditure' , 'company']])

model2 = LogisticRegression(C=100000 ,solver='newton-cg' , random_state=1234).fit(ex1_x , ex1_y)

predict2_proba = model2.predict_proba(ex1_x)

predict2 = model2.predict(ex1_x)
predict2[:3]

pd.DataFrame(predict2_proba.round(3) , columns=['other' , 'LL'])[:3]

# ## 1-3 로지스틱 회귀 분석 다항 분석 
#
# 회사 예측 하기  
# 로지스틱 회귀 분석시 타깃 항목값은 0 또는 1이었다. 만약 타깃 종류가 2가지가 아닌 3가지 이상이면 어떻게 해야 할까? 
#
# <br><b>종속변수</b> : car_type(SS/MM/LL) 
# <br><b>독립변수</b> : age, salary, expenditure, company(drop_first=True)
#
# 옵션값 - C=100000 , solver='newton-cg' 

ex2_y = basetable['car_type']
ex2_x = pd.get_dummies(basetable[['age','salary','expenditure','company']] , drop_first=True)


x_train1 , x_test1 , y_train1 , y_test1 = train_test_split(ex2_x , ex2_y , train_size=0.7 , random_state=123)

model3 = LogisticRegression(C=100000 , solver='newton-cg',multi_class='multinomial').fit(x_train1 , y_train1)

predict3 = model3.predict(x_test1)
predict3_proba = model3.predict_proba(x_test1)

predict3

pd.DataFrame(predict3_proba.round(3))

# ## Quiz 로지스틱 회귀분석 - 붓꽃 데이터 셋 활용 
# sklearn 라이브러리 활용을 통한 붓꽃 품종 분류 
#
# <br>1. 종속변수 species 중 virginica 인 여부를 분류하는 로지스틱 회귀모델 만들기 
# <br>2. 독립변수는 'sepal_length', 'sepal_width',	'petal_length',	'petal_width' 4개 변수로 하되 
#        정규화(StandardScaler)한 4개 변수를 활용할 것 
# <br>3. 트레이닝셋, 테스트셋 분류할 필요 없이 150개 샘플을 사용하고 150개 샘플 그대로 모델에 적용해 예측해 본다
# <br>4. 이때 virginica 예측 분류 모델 관련 재현율을 구하시오
# <br>( sklearn.liner_model LogisticRegression 활용 파라미터 값 C=100000 , random_state = 123, solver='newton-cg' 나머지 디폴트)

iris = pd.read_csv('data/iris.csv')

iris['species'] = np.where(iris['species'] == 'virginica' , 1 , 0)

iris_x = pd.DataFrame(StandardScaler().fit_transform(iris[[ 'sepal_length', 'sepal_width', 'petal_length', 'petal_width']])
                     , columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])

iris_y = iris['species']

model_iris = LogisticRegression(C=100000 , random_state = 123, solver='newton-cg').fit(iris_x , iris_y)

predict_iris = model_iris.predict(iris_x)

print(classification_report(iris_y, predict_iris ))

# ## 타이타닉
# <br>과제 정의) 
# <br>1. 종속변수 : survived / 독립변수 : sex, age, sibsp, parch, fare, embarked, class로 한다
# <br>2. 해당 변수 중 하나라도 결측치가 있는 데이터 셋은 제외한다 
# <br>3. 독립변수 중 sex, embarked, class는 더미변수화 하며 drop_first 옵션은 True로 지정한다 
# <br>4. 수치형 독립변수 중 가장 왜도가 큰 값은 fare 값은 log변환 한다.(변환시 np.log( 1+ 변수) 활용 할 것) 
# <br>5. 트레이닝셋:테스트셋 7:3으로 분할한다(random_state =123)  
# <br>6. sklearn 라이브러리 활용 로지스틱 회귀분석 트레이닝셋 학습을 진행한다(파라미터 값 C=100000 ,solver='newton-cg' 적용)
# <br>7. 학습한 모델을 바탕으로 테스트셋 예측을 진행한다. 
# <br>   이때 f1-score를 높이기 위해 예측값은 
#        확률값을 확인하여 target=1로 예측한 확률값이 0.4보다 큰 경우에는 1로 나머지는 0으로 분류한다 
#        
# <br><b> 7번의 단계를 모두 수행한 후 class 1에 대한 f1-score에 대해 서술 하시오</b>

titanic = pd.read_csv('data//seaborn_titanic.csv')

titanic_1 = titanic[['survived' , 'sex' , 'age' , 'sibsp' , 'parch' ,'fare','embarked' , 'class']].dropna().reset_index(drop=True)

titanic_2 = pd.get_dummies(titanic_1[['survived' , 'sex' , 'age' , 'sibsp' , 'parch' ,'fare','embarked' , 'class']], drop_first=True)

titanic_2['fare'] = np.log(1+titanic['fare'])

train , test = train_test_split(titanic_2 , train_size=0.7 , random_state=123)
train.shape

model_titanic = LogisticRegression(C=100000 ,solver='newton-cg').fit(train.drop('survived' , axis=1) , train['survived'])

predict_titanic = model_titanic.predict(test.drop('survived' , axis=1))

predict_titanic

predict_titanic_proba = model_titanic.predict_proba(test.drop('survived' , axis=1))

ans = np.where(predict_titanic_proba[ : , 1] > 0.4 , 1 , 0)
print(classification_report(ans , test['survived']))

df_ti = titanic.copy()

rdf_ti = df_ti.drop(['embark_town','deck'] , axis=1)

rdf_ti = rdf_ti.dropna(subset='age')

rdf_ti['embarked'] = rdf_ti['embarked'].fillna(rdf_ti['embarked'].value_counts().idxmax())

#독립변수
ndf = rdf_ti[['survived', 'pclass' , 'sex', 'age','sibsp','parch','fare','embarked']]

ndf = pd.get_dummies(ndf , columns=['sex','embarked'])

x = ndf.drop('survived' , axis=1)
y = ndf['survived']

st = StandardScaler().fit_transform(x)

x_st = pd.DataFrame(st , columns=x.columns)

x_train , x_test , y_train , y_test = train_test_split(x_st,y , train_size=0.7 , random_state=123)

from sklearn.svm import SVC

#분류모형 선택
model = SVC().fit(x_train , y_train)

predict = model.predict(x_test)

# +

print(classification_report(y_test , predict))
# -


