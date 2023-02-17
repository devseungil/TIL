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
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ## PCA 프로세스
# <br>1.정규화 -> 2. 공분산 행렬 계산 -> 3. 공분산 행렬 고유벡터와 고유값 계산 -> 4. 주성분 구하기
#
# ### 콘크리트 데이터 셋 활용, PCA 통해 만든 합성변수로 종속변수 strength을 예측하는 다중 회귀 분석 모델 설계
# <br>1030 rows × 9 columns

df = pd.read_csv('../data/yellowbrick_concrete.csv')

# +
# 다중공선성 확인
from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF
  
def feature_engineering_VIF(X):
    vif = pd.DataFrame()
    vif['VIF_Factor'] = [VIF(X.values, i) for i in range(X.shape[1])]
    vif['Feature'] = X.columns
    return vif


# -

feature_engineering_VIF(df).sort_values('VIF_Factor', ascending=False)

# VIF 지수가 10 초과하는 변수들이 다수 존재 
# <br> PCA를 통해 차원 축소, 복잡성을 줄이자!

st = StandardScaler()
df_st = pd.DataFrame(st.fit_transform( df.drop('strength',axis=1) ) , columns= df.drop('strength' , axis = 1).columns)
df_st

pca_model = PCA(random_state=123).fit(df_st)

df_pca = pd.DataFrame(pca_model.transform(df_st) , columns= ['pca_' + str(i) for i in range(1,9)])

df_pca

pca_model.explained_variance_ratio_.cumsum() #6개 까지만 해도 됨

import statsmodels.api as sm

eight_model = sm.OLS(df['strength'] , sm.add_constant(df_st)).fit()

# +
# eight_model.summary()
# -

six_model = sm.OLS(df['strength'] , sm.add_constant(df_pca.iloc[ : , :6])).fit()

# +
# six_model.summary()
# -

eight_model_predict = eight_model.predict(sm.add_constant(df_st))
six_model_predict = six_model.predict(sm.add_constant(df_pca.iloc[ : , :6]))

from sklearn.metrics import mean_squared_error

print(mean_squared_error(eight_model_predict , df['strength'])**0.5)
print(mean_squared_error(six_model_predict , df['strength'])**0.5)
print("변수가 2개 줄었어도 성능차이가 별로 없다")

# ## 4. Asociation rules
# 250명의 식료품 구매 이력을 바탕으로 연관성 분석 수행
# <br> (file: hkdataset_associaterules.csv - 식료품 데이터셋이라 명명)
#
# <br> A회사 임직원을 대상으로 연관성 규칙 확인 
# <br> 우유를 단일 선행으로 하는 규칙을 만들며 후행 품목 수는 상관없다.    
# <br> 이를 위해 A회사 100명의 식료품 구매 이력을 확인하여 A회사 임직원 대상으로 장바구니 분석을 수행한다.
# <br> 이때 우유를 선행으로 하는 규칙 중 Lift 값이 가장 높은 규칙은 무엇인지 확인하시오.
# <br> HINT: 식료품 데이터셋에서 250명의 식료품 구매 이력을 불러 온 후 A회사 정보만 필터링 한후 수행한다. 
# <br> 
# <br><b> 관련 라이브러리 및 하이퍼 파라미터 값 </b>
# <br> from mlxtend.preprocessing import TransactionEncoder
# <br> from mlxtend.frequent_patterns import apriori, association_rules
# <br> 조건 min_support=0.1, min_confidence=0.01

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

df2_1 = pd.read_csv('../data/hkdataset_associaterules.csv')

df2_2 = pd.read_csv('../data/hk_221206.csv')
df2_2 = df2_2.rename(columns = {'name':'id'})

df2= pd.merge( df2_1, df2_2[['id', 'company']], how='left', on='id' )

df2 = df2.loc[ df2['company'] == 'A' , : ]

df2

datatable = df2.groupby('id').apply( lambda x : x['item'].tolist() ).reset_index().rename( columns = {0:'item'} )

datatable

tr = TransactionEncoder()
tr_fit = tr.fit(datatable['item'])

tr_data = pd.DataFrame(tr.transform(datatable['item']) , columns= tr.columns_)
tr_data

# 지지도 10%이상만 찾아오기
ap = apriori(tr_data, use_colnames=True,  min_support=0.1)
ap

#신뢰도 0.01 이상만 주셈
fdf = association_rules( ap, metric='confidence', min_threshold=0.01) 
#fdf = association_rules( apri1, metric='lift', min_threshold=1.5) 향상도 1.5이상만 가져오셈

#antecedents 선행
# consequents 후행
fdf

fdf['ante_n'] = [ len(fdf['antecedents'][i]) for i in range(fdf.shape[0]) ] #선행 아이템의 개수

fdf['conse_n'] = [ len(fdf['consequents'][i]) for i in range(fdf.shape[0]) ] #후행 아이템의 개수

fdf

fdf['antecedents'][0]

fdf.loc[ (fdf['ante_n'] == 1) & (fdf['antecedents'] == frozenset({'Milk'}) ) , : ].sort_values('lift' , ascending = False)[:5]

# answer = 선행을 우유로 한 규칙중 제일 향상도가 높은 규칙은 우유를 사고 빵과 옥수수를 사는것이다


