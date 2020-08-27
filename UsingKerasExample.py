# 딥러닝 프레임 워크에는 Tensorflow도 있지만 Keras도 있다.
# Keras 설명서 : https://keras.io/

from keras.layers import Dense
from keras.models import Sequential

x_train = []
y_train = []

# Sequential 모델을 선언하면 add()라는 함수를 사용해 층들을 붙일 수 있다.
model = Sequential()

# Dense(각 층의 노드 수, 입력 데이터의 열의 수(입력 층에서만 명시), 활성함수의 종류)
model.add(Dense(12, input_dim=5, activation='sigmoid')) # 입력 층
model.add(Dense(30, activation='sigmoid'))              # 은닉 층
model.add(Dense(1, activation='sigmoid'))               # 출력 층

# 인공신경망을 학습시키기 위한 Optimizer 설정
# loss 함수 : Mean Squared Error, Optimizer(Gradient Descent) : RMSProp
model.compile(loss='mse', optimizer='RMSProp')

# 모델 학습

# 전체 학습 데이터에 대해 한번 모델을 업데이트 하는 방식
# x_train : 입력 데이터, y_train : 학습 목표 (타깃), epoch : 학습 데이터를 몇 번 사용해서 학습할 것인지
# model.fit(x_train, y_train, epoch=1)

# 전체 학습 데이터를 작은 단위로 쪼개서 여러 번에 걸쳐서 모델을 업데이트하는 방식 : mini-batch 방식
# batch : 전체 데이터를 쪼갠 작은 단위, batch_size : 몇 개의 데이터를 모아서 한번 모델을 업데이트할지
model.fit(x_train, y_train, batch_size=32, epoch=1)
