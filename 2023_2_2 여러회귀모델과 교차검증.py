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
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from sklearn.linear_model import LogisticRegression
# matplotlib에서 한글 폰트를 설정하는 방법
import matplotlib.pyplot as plt
plt.rc("font", family="malgun gothic")

# 음수 기호 출력 방법
import matplotlib
matplotlib.rcParams["axes.unicode_minus"]=False

# 그래프를 선명하게 출력하는 방법
# %config InlineBackend.figure_format = "retina"

import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
# -

housing = fetch_california_housing(as_frame=True)

x = housing.data
y = housing.target
x_train , x_test , y_train , y_test = train_test_split(x,y , train_size=0.7 , random_state=123)
train_set = pd.concat([x_train , y_train] ,axis=1)
test_set = pd.concat([x_test , y_test] , axis=1)

train_set


# +
def round_num(df) :
    lst = ['AveRooms','AveBedrms','AveOccup']
    for i in lst :
        df[i] = df[i].astype(int)
    return df

def z_exclude(df) :
    columns = ['MedInc','AveRooms','Population','AveOccup']
    for column in columns :
        mean = df[column].mean()
        std = df[column].std()
        z = abs(df[column] - mean) / std
        df = df[z < 2]
    return df

def classify(df) :
    if df < 600 :
        return 'few'
    elif df > 3000 :
        return 'many'
    else :
        return 'usually'

def summary(dataframe) :
    df = dataframe.copy()
    df = round_num(df)
    
    df = df[df['HouseAge'] < 52]
    df = df[df['MedHouseVal'] < 5]
    df = z_exclude(df)
    
    df['Population_feature'] = df['Population'].apply(classify)
    pop_dummy = pd.get_dummies(df['Population_feature'] , drop_first=True)
    df = pd.concat([df , pop_dummy] , axis=1)
    
    x = df.drop(['MedHouseVal' , 'Population_feature'] , axis = 1)
    y = df['MedHouseVal']
    return x,y


# -

x,y = summary(train_set)

x

sc = StandardScaler().fit(x)
x_sc = pd.DataFrame(sc.transform(x) , columns=x.columns)


from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

model_svr = SVR().fit(x_sc , y)

pred_svr = model_svr.predict(x_sc)

mse_svr = mean_squared_error(pred_svr , y)
mse_svr

rmse_svr = mse_svr ** 0.5
rmse_svr

model_rfr = RandomForestRegressor().fit(x_sc , y)
pred_rfr = model_rfr.predict(x_sc)
mse_rfr = mean_squared_error(pred_rfr , y)
rmse_rfr = mse_rfr ** 0.5
rmse_rfr

model_gbr = GradientBoostingRegressor().fit(x_sc , y)
pred_gbr = model_gbr.predict(x_sc)
mse_gbr = mean_squared_error(pred_gbr , y)
rmse_gbr = mse_gbr ** 0.5
rmse_gbr

msc = MinMaxScaler().fit(x)
x_m = pd.DataFrame(msc.transform(x) , columns=x.columns)

model_mlp = MLPRegressor().fit(x_m , y)
pred_mlp = model_mlp.predict(x_m)
mse_mlp = mean_squared_error(pred_mlp , y)
rmse_mlp = mse_mlp ** 0.5
rmse_mlp

ev = pd.DataFrame([rmse_svr , rmse_rfr , rmse_gbr , rmse_mlp] , index=['SVR' , 'RFR' , 'GBR' , 'MLP'] , columns=['RMSE평가'])
ev

score_svr = cross_val_score(model_svr , x_sc , y , scoring='neg_mean_squared_error' , cv=5)

rmse_score_svr = ((-score_svr) ** 0.5).mean()

score_rfr = cross_val_score(model_rfr , x_sc , y , scoring='neg_mean_squared_error' , cv=5)
rmse_score_rfr = ((-score_rfr) ** 0.5).mean()

score_gbr = cross_val_score(model_gbr , x_sc , y , scoring='neg_mean_squared_error' , cv=5)
rmse_score_gbr = ((-score_gbr) ** 0.5).mean()

score_mlp = cross_val_score(model_mlp , x_m , y , scoring='neg_mean_squared_error' , cv=5)
rmse_score_mlp = ((-score_mlp) ** 0.5).mean()

ev['CrossValidation'] = [rmse_score_svr , rmse_score_rfr , rmse_score_gbr , rmse_score_mlp]

ev

pip install jupytext

jupyter notebook --generate-config




