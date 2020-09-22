#1. 데이터
import numpy as np
x = np.array([range(100)])
y = np.array([range(201,301), range(301,401)])
x = np.transpose(x)
y = np.transpose(y)
print(x.shape)
print(y.shape)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
        x, y, random_state=66, test_size=0.4, shuffle=False
)
x_val, x_test, y_val, y_test = train_test_split(
        x_test, y_test, random_state=66, test_size=0.5, shuffle=False
)
print(x_test.shape)

#2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
# model.add(Dense(5, input_dim = 3, activation ='relu'))
model.add(Dense(5, input_shape = (1, ), activation ='relu'))
model.add(Dense(3))
model.add(Dense(4))
model.add(Dense(2))

#3. 훈련
# model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
# model.fit(x, y, epochs=100, batch_size=3)
model.fit(x_train, y_train, epochs=100, batch_size=1,
validation_data=(x_val, y_val))

#4. 평가 예측
loss, mse = model.evaluate(x_test, y_test, batch_size=1)
print("mse : ", mse)
y_predict = model.predict(x_test)
print(y_predict)

#RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

# R2 구하기
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test, y_predict)
print("R2 : ", r2_y_predict)
