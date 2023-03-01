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
# MNIST 데이터 활용 (딥러닝의 'hello world')
#
# fashion_mnist module : Fashion-MNIST dataset. , mnist module : MNIST handwritten digits dataset. 
#
# NIST : 손글씨 흑백숫자
#
#     0 ~ 9까지의 숫자를 예측하는 다중분류 문제
#     데이터 숫자 이미지(28,28)와 숫자에 해당하는 레이블로 구성돼있음
#     
# This is a dataset of 60,000 28x28 grayscale images of the 10 digits, along with a test set of 10,000 images
#     6만개의 학습데이터 , 10,000 개의 테스트 데이터
# -

import tensorflow as tf
import numpy as np

# 데이터 로드
from tensorflow.keras.datasets.mnist import load_data
(x_train, y_train), (x_test, y_test) = load_data(path = 'mnist.npz')

#데이터 형태 확인 꼭 확인하기 << 훈련데이터, 테스트데이터 , 검증용데이터(val)(훈련데이터에서 추출해서 구성) >>
print(x_train.shape , y_train.shape)
print(x_test.shape , y_test.shape)
print(type(x_train))
x_train.max()

import matplotlib.pyplot as plt

# +
# 임의데이터 추출 : 0 ~ 59999의 범위에서 무작위로 2개 정수를 추출하자
np.random.seed(777)
sample_size = 2

random_idx = np.random.randint(60000, size = sample_size) 

for idx in random_idx : # [47919 , 15931]
    img = x_train[idx, :]
    label = y_train[idx]
    plt.figure()
    plt.imshow(img)
    plt.title('%dth data, label is %d' % (idx,label) , fontsize=15)
    # 47919 번째 훈련데이터의 레이블은 5
# -

img = x_test[0, :]
label = y_test[0]
plt.figure()
plt.imshow(img)
plt.title('%dth data, label is %d' % (idx,label) , fontsize=15)

# 데이터 추출해서 내용확인해보자
for i in x_train[0]:
    for j in i :
        print('{:3}'.format(j),end='')
    print()

# +
# 훈련 , 테스트 임의의 비율 분리
#훈련데이터에서 검증데이터 분리
from sklearn.model_selection import train_test_split

x_train, x_val , y_train , y_val = train_test_split(x_train , y_train , test_size=0.3 , random_state=777)
print('훈련데이터 :',x_train.shape,'레이블 :',y_train.shape)
print(x_val.shape,y_val.shape)
# -

# ## 학습을 위한 전처리 수행 ( 학습을하지않으면 데이터폭이 넓어져서 안조음

# +

#전처리  ->  0 ~ 1 스케일조정, Dense층 사용하기 위해 784(28*28)차원에서 1차원 배열로 변환 (인풋셰입은1차원)

num_x_train = x_train.shape[0]  # 42000 , 28 , 28 의 0번 인덱스 = 42000
num_x_val = x_val.shape[0]      # 18000 , 28 , 28
num_x_test = x_test.shape[0]    # 10000  ,28 , 28

# 전처리 스케일 조정
x_train = (x_train.reshape((num_x_train , 28*28))) /255 # x 데이터가 0~255 라서 255로 나눈것
x_val = (x_val.reshape((num_x_val , 28*28))) / 255
x_test = (x_test.reshape((num_x_test , 28*28))) / 255
# -

print(x_train.shape)# 모델 데이터 입력을 위해 데이터를 784차원으로 변경
x_train.max()

# +
# softmax함수를 사용하려면 범주형 레이블로 변환해야함 -> to_categorical() / 각 데이터의 레이블을 범주형으로변경
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)
y_test = to_categorical(y_test)

print(y_train)
y_train.shape
# -

from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import mse


# +
model = Sequential()

# 784차원의 데이터를 입력받고 64개의 출력을 가지는 첫번째 히든레이어 Dense 층 h1
model.add(Dense(64, input_shape=(784,) , activation = 'relu')) 

# 32개의 출력을 가지는 히든레이어 Dense 층 h2
model.add(Dense(32, activation = 'relu')) 

# 10개의 출력을 가지는 아웃풋레이어 
model.add(Dense(10, activation = 'softmax')) #10개 출력을 가지고 있는 신경망 (y가 10개 0~9)

# metrics =['acc'] 모니터링 할 평가지표
model.compile(optimizer= 'adam' ,loss = 'categorical_crossentropy' , metrics =['acc'] ) 

#학습시키기 epochs (훈련반복횟수)
# validation_data에 검증데이터셋을 전달하고 128의 배치 크기를 사용하고 전체데이터를 30회 반복해서 학습(가중치 다르게 주면서)
#훈련데이터 / 배치사이즈 = 가중치업데이트 횟수 (속도향상) 배치사이즈안하면 매번 가중치업데이트

history = model.fit(x_train, y_train, epochs= 30 , batch_size = 128, validation_data=(x_val,y_val))

model.evaluate(x_test, y_test)

result = model.predict(x_test)
print(result.shape)
print(result[0])

# +
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss)+1)
fig = plt.figure(figsize = (10,5))

ax1 = fig.add_subplot(1,2,1)
ax1.plot(epochs, loss, color='blue', label = 'train_loss')
ax1.plot(epochs, val_loss, color='orange' , label = 'val_loss')
ax1.set_title('train_val loss')
ax1.set_xlabel('epochs')
ax1.set_ylabel('loss')
ax1.legend()

acc = history.history['acc']
val_acc = history.history['val_acc']

ax1 = fig.add_subplot(1,2,2)
ax1.plot(epochs, acc, color='blue', label = 'train_acc')
ax1.plot(epochs, val_acc, color='orange' , label = 'val_acc')
ax1.set_title('train_val acc')
ax1.set_xlabel('epochs')
ax1.set_ylabel('acc')
ax1.legend()

# + active=""
# 두 그래프가 어디서 부터 벌어지는지 확인
# 1) 과대적합 문제가 발생
# 2) 데이터 특성, 모델 구조등을 수정해보고 재학습
# 3) 벌어지기 전 까지의 모델을 사용하여 결과를 확인하고 저장 및 기록
# -

# 모델평가
model.evaluate(x_test , y_test)

# +
#학습된 모델을 통해 값 예측
np.set_printoptions(precision = 7) # 소수점 제한
result = model.predict(x_test)

print(result.shape)
print(f'각 클래스에 속할 확률 : {result[0]}')
# -

arg_result = np.argmax(result, axis = -1) # 가장 큰값의 인덳스 값 리턴 argmax
plt.imshow(x_test[0].reshape(28,28))
plt.title('predicted first img:' + str(arg_result[0]));
arg_result

# +
#모델평가 방법 1 혼동행렬
from sklearn.metrics import classification_report , confusion_matrix
import seaborn as sns

plt.figure(figsize = (5,5))
cm = confusion_matrix(np.argmax(y_test, axis = -1) , np.argmax(result, axis = -1) )
sns.heatmap(cm, annot = True , fmt = 'd' , cmap = 'Blues')
plt.xlabel('predicted label' , fontsize=10)
plt.ylabel('true label' , fontsize=10)
# -

#모델평가 방법 2 분류보고서
print(classification_report(np.argmax(y_test, axis = -1) , np.argmax(result, axis = -1)))


