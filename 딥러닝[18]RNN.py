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

# ## RNN(Recurrent Neural Network) : 순환 신경망
# #### 순서가 있는 시퀀스 데이터, time series data(시계열 데이터)를 입력하여 예측 , 주가데이터 , 음성데이터

# + active=""
# SimpleRNN 의 Weight와 bias의 shape
# - N : batch_size , T : Sequence_length , D : input_dim , H : hidden_size
# - X_all : (N , T , D)
# - X_1 : (N , D)
# - Wx : (D , H)
# - Wh : (H , H) h_prev : (N,H) 이기 때문이다.
# - b : (H)
# -

import tensorflow as tf
import numpy as np



# ### One cell: 4 (input_dim) in 2 (hidden_size)

# ![image](https://cloud.githubusercontent.com/assets/901975/23348727/cc981856-fce7-11e6-83ea-4b187473466b.png)

# + active=""
# << 그림설명 >>
# 시간(t)에서의 상태를 고려할때 시간(t)인 입력인 X(t)이외의 시간 (t-1)의 상태를 나타내는 h(t-1)을 유지하면서
# 시간(t)에게 리턴한다.
# [핵심] 이전 계산에 따라 출력이 문자열의 모든요소에 대해 동일한 작업을 수행한다. 이전에 계산된 정보를 기억할수 있다
#
# ex) 주가 10년의 데이터의 과거시점부터(순전파), 현재시점(역전파)
# <중요>
# 문장의 경우 단어를 차례로 특징량의 숫자의 열로 계산해간다 -> 전 후의 단어의 특징량의 숫자를 연결한다 ->
# 모든단어를 구문트리로 만든다 -> 숫자만 연결하면 숫자의 개수가 점점 커지게된다 -> 가중치w 결합 -> 하나의 특징량의 숫자로 출력
# -

h = [1, 0, 0, 0]
e = [0, 1, 0, 0]
l = [0, 0, 1, 0]
o = [0, 0, 0, 1]

# +
#2 모델을 실행해서 결과를 리턴
# 하나의 셀을 RNN적용  input_dim(4) - > output_dim(2)
x_data = np.array([[h]], dtype=np.float32)

hidden_size = 2
cell = tf.keras.layers.SimpleRNNCell(units=hidden_size)
rnn = tf.keras.layers.RNN(cell,return_sequences=True,return_state=True)
# return_sequences=True : 출력으로 시퀀스 전체를 리턴
outputs,states = rnn(x_data)

# h = [1, 0, 0, 0]
print('x_data: {}, shape: {}'.format(x_data, x_data.shape)) # X : [N, T, D]
print('outputs: {}, shape: {}'.format(outputs, outputs.shape)) #OUT : [N, T, H]
print('states: {}, shape: {}'.format(states, states.shape))   #STATE : [N, H]

# +
#3 2번의 모델을 함축으로 실행해서 결과 리턴
rnn = tf.keras.layers.SimpleRNN(units=hidden_size,return_sequences=True,
                                return_state=True)
outputs,states = rnn(x_data)

print('x_data: {}, shape: {}'.format(x_data, x_data.shape))    # X : [N, T, D]
print('outputs: {}, shape: {}'.format(outputs, outputs.shape)) #OUT : [N, T, H]
print('states: {}, shape: {}'.format(states, states.shape))    #STATE : [N, H]
# -

# ### Unfolding to n sequences

# ![image](https://cloud.githubusercontent.com/assets/901975/23383634/649efd0a-fd82-11e6-925d-8041242743b0.png)

# +
x_data = np.array([[h,e,l,l,o]], dtype=np.float32)

hidden_size = 2
rnn = tf.keras.layers.SimpleRNN(units=hidden_size,return_sequences=True,
                                return_state=True)
outputs,states = rnn(x_data)

print('x_data: {}, shape: {}'.format(x_data, x_data.shape))    
print('outputs: {}, shape: {}'.format(outputs, outputs.shape))
print('states: {}, shape: {}'.format(states, states.shape))   
# -

# ### Batching input

# ![image](https://cloud.githubusercontent.com/assets/901975/23383681/9943a9fc-fd82-11e6-8121-bd187994e249.png)

# +
x_data = np.array([[h, e, l, l, o],
                   [e, o, l, l, l],
                   [l, l, e, e, l]], dtype=np.float32)

hidden_size = 2
rnn = tf.keras.layers.SimpleRNN(units=hidden_size,return_sequences=True,
                                return_state=True)
outputs,states = rnn(x_data)

print('x_data: {}, shape: {}'.format(x_data, x_data.shape))    
print('outputs: {}, shape: {}'.format(outputs, outputs.shape)) 
print('states: {}, shape: {}'.format(states, states.shape))   

# +
import matplotlib.pyplot as plt

x = np.arange(-200,200)
x = x/10
y = np.tanh(x)

plt.plot(x,y)
plt.grid(True)
print(np.tanh(19))
print(np.tanh(20))
# -


