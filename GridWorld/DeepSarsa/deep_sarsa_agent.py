import copy
import pylab
import random
import numpy as np
from environment import Env
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential

EPISODES = 2500

class DeepSARSAAgent:
    def __init__(self):
        self.load_model = False

        # 가능한 모든 행동
        self.action_space = [0, 1, 2, 3, 4]

        # 상태의 크기와 행동의 크기
        self.action_size = len(self.action_space)
        self.state_size = 15
        self.discount_factor =0.99
        self.learning_rate = 0.001

        self.epsilon = 1
        self.epsilon_decay = .9999
        self.epsilon_min = 0.01

        # build_model은 처음 agent를 호출할 때 한 번 실행
        self.model = self.build_model()

        if self.load_model:
            self.model.load_weights('./save_model/deep_sarsa_trained.h5') # 이거는 뭘까

    # 상태가 입력, 큐함수가 출력인 인공신경망
    def build_model(self):
        model = Sequential()

        model.add(Dense(30, input_dim=self.state_size, activation='relu'))
        model.add(Dense(30, activation='relu'))

        # 출력으로 나오는 게 큐함수인데 큐함수는 0과 1 사이의 값이 아니기 때문에 출력층의 활성함수는 선형함수
        model.add(Dense(self.state_size, activation='linear'))

        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            state = np.float32(state)

            # state를 모델에 집어넣어서 출력을 반환
            q_values = self.model.predict(state)

            # 출력이 [[],[],[],[],[]]형식이기 때문에 [0]을 붙여서 한꺼풀 벗김, 5개(행동의 개수)의 큐함수
            return np.argmax(q_values[0])

    def train_model(self, state, action, reward, next_state, next_action, done):

        # 딥살사는 입실론을 계속 감소 시킴 : 초반에는 탐험을 자주 하고, 후에는 탐험을 줄임 -> 예측대로 움직임
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # 케라스에 들어가는 입력은 float 이어야 함
        state = np.float32(state)
        next_state = np.float32(next_state)
        target = self.model.predict(state)[0]

        if done:
            target[action] = reward
        else:   # 살사의 큐함수 업데이트 식에서 정답(타깃) : R + r*Q(s',a')
            target[action] = (reward + self.discount_factor *
                              self.model.predict(next_state)[0][next_action])
            # 실제로 수행한 행동에 해당하는 큐함수에 대해서만 오류함수를 계산해야함^^^^^^^^^^^^, 다른 4개의 큐함수는 오차가 0

        # 계산한 타깃과 상태 입력으로 인공신경망을 업데이트
        # 출력값 reshape : model.predict(state)의 출력 형태로 변형
        target = np.reshape(target, [1, 5])

        # 인공신경망 업데이트
        self.model.fit(state, target, epochs=1, verbose=0)


if __name__ == "__main__":
    env = Env()
    agent = DeepSARSAAgent()

    global_step = 0
    scores, episodes = [], []

    for e in range(EPISODES):
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, 15])

        while not done:
            global_step += 1

            # 1. 현재 상태에 대한 행동 선택
            action = agent.get_action(state)

            # 2. 행동을 취한 후 환경에서 한 타임스텝 진행, 3. 환경으로부터 다음 상태와 보상을 받음
            next_state, reward, done = env.step(action)

            # reshape는 데이터는 그대로 두고 차원만을 바꿔주는 함수
            next_state = np.reshape(next_state, [1, 15])

            # 4. 다음 상태에 대한 행동을 선택
            next_action = agent.get_action(next_state)

            # 5. 환경으로부터 받은 정보를 토대로 학습을 진행
            # 기존의 큐함수 테이블 대신 인공신경망을 사용함, 입력 : 상태의 특징들, 출력 : 각 행동에 대한 큐함수(근사된 거)
            agent.train_model(state, action, reward, next_state, next_action, done)

            state = next_state
            score += reward
            state = copy.deepcopy(next_state)

            if done:
                # 에피소드마다 학습 결과 출력
                scores.append(score)
                episodes.append(e)
                pylab.plot(episodes, scores, 'b')
                pylab.savefig("./save_graph/deep-sarsa.png")
                print("episode:", e, "score:", score, "global_step", global_step, "epsilon:", agent.epsilon)

        # 100 에피소드마다 모델 저장
        if e % 100 == 0:
            agent.model.save_weights("./save_model/deep_sarsa.h5")