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
# optimizer = 확률일땐 SGD(), 'sgd' , 아니면 'adam' 많이쓰임 

# + active=""
# res = model.predict(x_test)
# res_labels = np.argmax(res , axis =-1)
#
# print(res_labels[:10]) #예측데이터
#
# print(y_test[:10]) #실제데이터
# -

# ## 1) Sequential API

# +
# 1) Sequential API
from tensorflow.keras.layers import Activation
# case 1 :
model = Sequential()
model.add(Dense(512 , input_shape = (1,) , activation = 'relu'))
model.add(Dense(256 ,activation = 'relu'))
model.add(Dense(2 , activation = 'softmax'))

# case 2 :
model = Sequential([
    Dense(512, input_shape = (1,) , activation = 'relu'),
    Dense(256 ,activation = 'relu'),
    Dense(2 , activation = 'softmax')
])

# case 3 : 전이학습? 할때 필요하여 익숙해지기
model = Sequential()
model.add(Dense(512, input_shape = (1,)))
model.add(Activation('relu'))

model.add( Dense(256) )
model.add( Activation('relu') )

model.add( Dense(2) )
model.add( Activation('softmax') )
keras.utils.plot_model(model, show_shapes=True)
# -

# ## Functional API

# +
# 2 ) Functional API : 레이어를 함수 호출처럼 연결하면서 모델 구축
# 레이어가 분기되거나 통합되는 복잡한 모델을 만들 수 있다.
from tensorflow.keras import Model , Input

input = tf.keras.Input( shape = (1,) )
h = Dense(512 ,activation = 'relu')(input)
h = Dense(256 ,activation = 'relu')(h)
output = Dense(2 ,activation = 'softmax')(h)

model = Model(inputs = input , outputs = output)

# +
x_train = np.random.randint(0,5,(20,1))

y1_train = np.where(x_train%2 == 0 , [0,1] , [1,0])

y2_train = np.where(x_train%2 == 0 , [1,0] , [0,1])

input = tf.keras.Input( shape = (1,) )
h = Dense(512 ,activation = 'relu')(input)
h = Dense(256 ,activation = 'relu')(h)

# 분기작업 y1 , y2
output1 = Dense(2 ,activation = 'softmax')(h)
output2 = Dense(2 ,activation = 'softmax')(h)

#통합작업 : model , compile , fit
model = Model(inputs = input , outputs = [output1, output2])
                                         # y값이 2개라 로스값 2개 (y개수에 맞춰줘야함)
model.compile(optimizer = 'adam' , loss = ['categorical_crossentropy','categorical_crossentropy'] , metrics = ['acc'])

model.fit(x_train , [y1_train, y2_train] , epochs = 400 , batch_size = 32 , validation_split=0.2 , verbose = 1)
# -

keras.utils.plot_model(model, show_shapes=True)


# ## Subclassing API

# +
class My_model(Model):
    def __init__(self): 
        super(My_model , self).__init__() # [Model() <- My_model() 상속]
        
        self.dense1 = Dense(512 ,input_shape = (1,) , activation = 'relu')
        self.dense2 = Dense(256 , activation = 'relu')
        self.dense3 = Dense(2 , activation = 'softmax')
        
    def call(self , x , training = False) : 
        x = self.dense1(x)
        x = self.dense2(x)
        output = self.dense3(x)
        return output
    
model = My_model()

# +
x_train = np.random.randint(0,5,(20,1))
y_train = np.where(x_train%2 == 0 , [0,1] , [1,0])

model.compile(optimizer = 'adam' , loss = 'categorical_crossentropy' , metrics = ['acc'])

model.fit(x_train , y_train , epochs = 300 , batch_size = 8 , validation_split=0.2 )
