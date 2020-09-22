#1. 데이터
import numpy as np
x1 = np.array([range(100), range(311,411), range(100)])
x2 = np.array([range(101,201), range(311,411), range(101,201)])
y = np.array([range(501,601)]) #, range(711,811), range(100)]

x1 = np.transpose(x1)
y = np.transpose(y)
x2 = np.transpose(x2)

from sklearn.model_selection import train_test_split
x1_train, x1_test, y_train, y_test = train_test_split(
x1, y, random_state=66, test_size=0.4, shuffle=False
)
x1_val, x1_test, y_val, y_test = train_test_split(
x1_test, y_test, random_state=66, test_size=0.5, shuffle=False
)
x2_train, x2_test = train_test_split(
x2, random_state=66, test_size=0.4, shuffle=False
)
x2_val, x2_test= train_test_split(
x2_test, random_state=66, test_size=0.5, shuffle = False
)

#2. 모델 구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input
# model = Sequential()

input1 = Input(shape=(3,))
dense1 = Dense(100, activation='relu')(input1)
dense1_2 = Dense(30)(dense1)
dense1_3 = Dense(7)(dense1_2)

input2 = Input(shape=(3,))
dense2 = Dense(50, activation='relu')(input2)
dense2_2 = Dense(7)(dense2)

from keras.layers.merge import concatenate
merge1 = concatenate([dense1_3, dense2_2])

from keras.layers.merge import Concatenate
merge1 = Concatenate()([dense1_3, dense2_2])

model1 = Dense(10)(merge1)
model2 = Dense(5)(model1)
output = Dense(1)(model2)

model = Model(inputs = [input1, input2], outputs = output)
model.summary()

#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit([x1_train, x2_train], y_train,
epochs=100, batch_size=1,
validation_data=([x1_val, x2_val] , y_val))

#4. 평가 예측
mse = model.evaluate([x1_test, x2_test],
y_test, batch_size=1)
print("mse : ", mse)

y_predict = model.predict([x1_test, x2_test])
for i in range(len(y_predict)):
    print(y_test[i], y_predict[i])