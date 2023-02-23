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

# + active=""
# 텐서의 개념, 기본연산
# Tensor : 텐서의 기본자료형(int, float 같은) , 형태는 numpy 배열 형식, 차원은 Rank로 표현
#     사칙연산
#         생성 : tf.constant() -> 넘파이변환 numpy() -> 텐서로변환 tf.convert_to_tensor() 
#         연산 : tf.add(), subtract() , multiply() , divide() 서로 차원이같아야 연산가능
#         
# 활성화함수 ( Dense(activation= '') )
#
#     역할 : 비선형함수 , 데이터 폭을 조절해줌
#     
#         시그모이드 : 이진분류 , 로지스틱회귀(2개중 하나선택 1,0 T,F ) 라벨이 2 종류
#         
#         소프트맥스함수 : 라벨이 3종류 이상 중에 하나를 골라서 출력 = 다중클래스 분류
#         
#     *단층 퍼셉트론 = 이진분류 = 선형( and, or , not ) = 직선
#     
#     *다층 퍼셉트론 = 비선형(xor) = 곡선 = 심층 신경망 = 은닉층(히든레이어) 2개이상인 신경망
#     
# 경사 하강법

# + active=""
# << 모델 선언 방법 >>
# 1)tf.keras.models
#
#     class Model: Model groups layers into an object with training and inference features.
#
#     class Sequential: Sequential groups a linear stack of layers into a tf.keras.Model.
#     
# 2) Define - and - run : 신경망 모델 구성을 정의 한 후 데이터를 입력하는 방법
#     
#     - Sequential : 입력과 출력이 반드시 하나씩의 네트워크 구성밖에 정의 할 수 없다
#                    히든레이어 층을 선형 스택 원형으로 쌓아서 지정하기때문에 히든레이어 계층에서 네트워크를 분산 시킬수 없다
#       
#     - Functional API : 사용자 정의 네트워크 구성
#
# -

# # 이건 암기

# + active=""
# 손실함수 ( 꼭 암기 )
#
#     model.compile(optimizer=RMSprop() ,loss = mse , metrics =['acc'] )
#     
#     이진분류 : binary_crossentropy
#     
#     다중클래스 단일 라벨(정답)로 분류    :  categorical_crossentropy n개중의 하나의정답
#     
#     다중클래서 다중 레이블로 분류  :  binary_crossentropy n개중의 여러개정답
#     
#     회귀문제 (임의의 값) 정답없음  :  mse
#     
#     회귀문제 (0 , 1 or 0 ~ 1)      :  mse / binary_crossentropy ( 0 , 1 )
#     
# 활성화 함수
#     이진 분류 : sigmoid
#     
#     다중클래스로 분류 : softmax
#     
#     회귀 : 항등함수 아무거나
# ---------------------------------------------------------------------------
#             분류 (이진)                         분류( 다중클래스)                    회귀
# 활성화함수: sigmoid                               softmax                      없거나 항등함수(ex : 'linear')
# 손실함수:   binary_crossentropy                categorical_crossentropy               mse (평균제곱오차)
#
# optimizer = 확률일땐 SGD() , 아니면 'adam' 많이쓰임 
# -

# ---------------------------------

# + active=""
# 학습수행 (입력데이터, 정답라벨 , 에폭수 , 배치사이즈)
#
#  History.history = model.fit(x_train, y_train, epochs= 30 , batch_size = 128, validation_data=(x_val,y_val))
#     
#     배치사이즈를 작게하면 소비되는 메모리는 적음 , 연동에 적용되지 않거나 느리다
#     History.history : loss , val_loss , acc , val_acc
#     
# 사용방법(추출) : 
# model.history['loss']
# model.history['val_loss']

# + active=""
# 모델 평가
# model.evaluate(x_test, y_test)
# (입력데이터 , 출력데이터 , batch_size=None , sample_weight=None 가중치 , steps=None 샘플배치 평가종료를 선언하기 전의 총 단계 수

# + active=""
# 모델 저장
#  - model.save(filepath)
#  ex ) model.save('abc.h5')
#       model = load_model('abc.h5')
#       
#  가중치 저장, 로드
#  - import h5py  :  _hdf5 의 내용을 확인하는 모듈 (가중치 로드하려면 임포트 해줘야함)
#  - model.save_weights('abc.hdf5')
#  - model = load_weights('abc.hdf5')
#  
#  - tf.keras.regularizers : 가중치 정규화생성 / Dense() 안에 속성으로 줄 수도 있다.
#  
#  - Dense(16, input_shape=(10000,) , activation = 'relu' , kernel_regularizer= regularizers.l2(0.001) )
#      : 해당 계층의 가중 행렬의 계수마다 네트워크의 전체 손실에 0.001 * 가중치수식 을 더함

# +
# pip install tensorflow
# -

import tensorflow as tf
import numpy as np

tf.__version__

a = tf.constant([[1,2,3],[4,5,6]])
print(tf.rank(a)) # 2차원
a

#텐서를 생성하고 사칙연산
a = tf.constant(3)
b = tf.constant(2)
print(tf.add(a,b).numpy()) # 
print(tf.subtract(a,b).numpy() , type(tf.subtract(a,b).numpy()))
print(tf.multiply(a,b).numpy())
print(tf.divide(a,b).numpy() , type(tf.divide(a,b).numpy()))

c_square = np.square(c, dtype = np.float32)

# +
# 넘파이 배열 변환하고 다시 텐서로 변환하자 convert_to_tensor()
c = tf.add(a,b).numpy() # a,b의 합을 numpy 배열 형태로 변환한것
c_square = np.square(c , dtype = np.float32) #넘파이 제곱함수

c_tensor = tf.convert_to_tensor(c_square)

print(f"{c} -> {c_square} -> {c_tensor}")
print(type(c_square))
print(type(c_tensor))
# -

#사칙연산을 차원을 만들어서 생성
a = tf.constant([1,2])
b = tf.constant([3,4])
print(tf.add(a,b).numpy()) # 
print(tf.subtract(a,b).numpy() , type(tf.subtract(a,b).numpy()))
print(tf.multiply(a,b).numpy())
print(tf.divide(a,b).numpy() , type(tf.divide(a,b).numpy()))

a = tf.constant([[1,2],[3,4]])
b = tf.constant([[1,2],[3,4]])
print(tf.add(a,b)) # 
print(tf.subtract(a,b).numpy() , type(tf.subtract(a,b).numpy()))
print(tf.multiply(a,b).numpy())
print(tf.divide(a,b).numpy() , type(tf.divide(a,b).numpy()))

# +
c = tf.add(a,b).numpy() # a,b의 합을 numpy 배열 형태로 변환한것
c_square = np.square(c, dtype = np.float32) #넘파이 제곱함수

c_tensor = tf.convert_to_tensor(c_square)

print(f"{c} -> {c_square} -> {c_tensor}")

print(type(c_square))
print(type(c_tensor))

# +
from IPython.display import Image

Image('화면 캡처 2023-02-22 152233.png')
# -

# ### 단층 퍼셉트론 적용 알고리즘의 프로세스를 구현
#     - 순차모형 : tf.keras.models.Sequential
#     - 레이어 구성 : tf.keras.layers.Dense
#     - f(net) : Dense( activation = 'linear' ) 
#     - 경사 하강법 개선 : 확률적 경사하강법(SGD) 일부 데이터만 임의 추출해서 미분 tf.keras.optimizers.SGD
#     - loss : mse  tf.keras.losses.mse
#     - 평가지표 : 정답률 acc
#     - 모델구성 -> 레이어 추가 add() > 컴파일(모델준비) compile() >  학습 fit()  > 

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import mse


tf.random.set_seed(777)

# ## 단층 퍼셉트론
# model.add(Dense(1, input_shape=(2,) , activation = 'linear'))
#

# +


# 데이터준비 ( or 게이트 )
data = np.array([ [0,0], [1,0], [0,1], [1,1] ]) # x1, x2 특성을 가진 데이터가 4개
label = np.array([ [0], [1], [1], [1] ])
#모델생성
model = Sequential()
#주요 메서드
    #compile() , evaluate() , fit() , get_layer() , predict() , save() , summary()

#모델에 히든 레이어 추가 -> 두개의 특성을 가진 1차원 데이터를 입력받고 한개의 출력을 가지는 Dense 층
#Dense(1 => 하나의 히든 레이어(은닉계층)를 만들겠다
#은닉계층이 1개이고 입력계층 단위계수가 2개인것을 [퍼셉트론] 이라고한다
model.add(Dense(1, input_shape=(2,) , activation = 'linear')) #단층 퍼셉트론을 구성 = 퍼셉트론 생성 

# 모델준비
    #경사 하강법 개선에 대한 속성(optimizers), 로스, 정답률
    # metrics는 평가지표를 전달하면서 지정된함수를 []로 지정한다 (여러개가능)
model.compile(optimizer=SGD() ,loss = mse , metrics =['acc'] ) #대문자는 클래스명() 붙여주기

#학습시키기 epochs (훈련반복횟수)
model.fit(data, label, epochs= 200)

# -

#평가지표, 가중값확인, 정보확인
model.get_weights() #모델 가중치 확인

model.evaluate(data , label)

preds = model.predict(data)
for a,b in zip(preds , label):
    print(f'정답{b}될 예측값 {a}')

# ## 다층 퍼셉트론 적용 알고리즘의 프로세스를 구현
#
#     - 순차모형 : tf.keras.models.Sequential
#     - 레이어 구성 : tf.keras.layers.Dense
#     - f(net) : Dense( activation = 'relu' ) Dense( activation = 'sigmoid' )
#     - 경사 하강법 개선 : RMSProp , Adagrad  tf.keras.optimizers.RMSprop
#     - loss : mse  tf.keras.losses.mse
#     - 평가지표 : 정답률 acc
#     - 모델구성 -> 레이어 추가 add() > 컴파일(모델준비) compile() >  학습 fit()  > 

# + active=""
# 특징데이터 2개(인풋셰입) , 출력값(y) 3개로 하겠다
#
# model.add(Dense(10?, input_shape=(2,) , activation = '..')) 
# model.add(Dense(3, activation = '.')) 
# -

from tensorflow.keras.optimizers import RMSprop

# +
# 데이터준비
data = np.array([ [0,0], [1,0], [0,1], [1,1] ]) # x1, x2 특성을 가진 데이터가 4개
label = np.array([ [0], [1], [1], [0] ]) # xor 덴스2개
#모델생성
model = Sequential()
#주요 메서드
    #compile() , evaluate() , fit() , get_layer() , predict() , save() , summary()

#모델에 히든 레이어 추가 -> 
#Dense(32 

model.add(Dense(32, input_shape=(2,) , activation = 'relu')) #다층 퍼셉트론을 구성
model.add(Dense(1, activation = 'sigmoid')) #마지막엔 1개층으로해야 값이 나옴

# 모델준비
    #경사 하강법 개선에 대한 속성(optimizers), 로스, 정답률
    # metrics는 평가지표를 전달하면서 지정된함수를 []로 지정한다 (여러개가능)
model.compile(optimizer=RMSprop() ,loss = mse , metrics =['acc'] ) #대문자는 클래스명() 붙여주기

#학습시키기 epochs (훈련반복횟수)
model.fit(data, label, epochs= 100)
# -

# 모델의 구성도 확인
model.summary()

#평가 확인
model.evaluate(data, label)

result = model.predict(data)
print(result)

# ## 활성화함수
#     - 시그모이드 : 값의 범위가 0~1

import math
import matplotlib.pyplot as plt

# +
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import mse
from tensorflow.keras.optimizers import RMSprop

model = Sequential()
model.add(Dense(4,input_shape=(3,)))
model.add(Dense(4)) # 8의 n승 ?
model.add(Dense(1))
# -

from tensorflow.keras.models import *
inputs = tf.keras.Input(shape=(3,))
hidden1 = Dense(4)(inputs)
hidden2 = Dense(4)(hidden1)
outputs = Dense(1)(hidden2)
model = Model(inputs = inputs , outputs = outputs)


