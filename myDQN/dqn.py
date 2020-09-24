from keras.layers.convolutional import Conv2D
from keras.layers import Dense, Flatten
from keras.optimizers import RMSprop
from skimage.color import rgb2gray
from skimage.transform import resize
from keras.models import Sequential
from keras import backend as K
from collections import deque
import tensorflow as tf
import numpy as np
import random
import gym

# 몇개의 에피소드가 지난 후 학습을 진행할 지 설정
EPISODES = 50000

class DQNAgent:
    def __init__(self, action_size):
        self.render = False
        self.load_model = False

        self.state_size = (84, 84, 4)
        self.action_size = action_size

        # DQN 하이퍼 파라미터
        self.discount_factor = 0.99
        # self.learning_rate = 0.001

        self.epsilon = 1.
        self.epsilon_start, self.epsilon_end = 1.0, 0.1
        # self.epsilon_decay = 0.999
        # self.epsilon_min = 0.01
        self.exploration_steps = 1000000.
        self.epsilon_decay_step = (self.epsilon_start - self.epsilon_end)/self.exploration_steps

        self.batch_size = 32
        self.train_start = 50000
        self.update_target_rate = 10000

        # 리플레이 메모리! 최대 크기는 400000
        # deque로 일정한 크기의 메모리를 생성할 수 있다.
        self.memory = deque(maxlen=400000)
        # 브레이크 아웃에서 처음에 agent가 빠지는 오류를 방지해 아무것도 하지 않는 구간 무작위로 설정, 그 무작위의 상한선
        self.no_op_steps = 30

        # 모델과 타깃 모델 생성
        self.model = self.build_model()
        self.target_model = self.build_model()
        # 타깃 모델 초기화
        self.update_target_model()

        self.optimizer = self.optimizer()

        # 텐서보드 설정
        self.sess = tf.InteractiveSession()
        K.set_session(self.sess)

        self.avg_q_max, self.avg_loss = 0, 0
        # 미리 변수들에 대한 형태를 생성함
        self.summary_placeholders, self.update_ops, self.summary_op = self.setup_summary()
        # 변수들을 저장
        self.summary_writer = tf.summary.FileWriter('summary/breakout_dqn', self.sess.graph)
        # 모든 형태를 지정한 후 다음 코드를 통해 변수를 초기화함
        self.sess.run(tf.global_variables_initializer())

        if self.load_model:
            self.model.load_weights("./save_model/breakout_dqn.h5")

    # Huber Loss를 이용하기 위해 최적화 함수를 직접 정의 : -1,1범위에서는 2차함수, 나머지 구간은 1차
    # REINFORCE 에서 했던 거 참고
    def optimizer(self):
        a = K.placeholder(shape=(None,), dtype='int32')
        y = K.placeholder(shape=(None,), dtype='float32')

        prediction = self.model.output

        # Huber Loss 구현 부분
        a_one_hot = K.one_hot(a, self.action_size)
        q_value = K.sum(prediction * a_one_hot, axis=1)
        error = K.abs(y - q_value)

        quadratic_part = K.clip(error, 0.0, 1.0)
        linear_part = error - quadratic_part
        loss = K.mean(0.5 * K.square(quadratic_part) + linear_part)

        # 경사하강법으로 RMSprop, 학습속도 0.00025
        optimizer = RMSprop(lr=0.00025, epsilon=0.01)
        # 각 가중치에 대한 업데이트 값을 구함
        updates = optimizer.get_updates(self.model.trainable_weights, [], loss)
        # 가중치를 업데이트하는 함수, train_model에서 self.optimizer를 호출하면 모델을 업데이트함
        train = K.function([self.model.input, a, y], [loss], updates=updates)

        return train

    # 상태가 입력, 큐함수가 출력인 인공신경망 생성
    def build_model(self):
        model = Sequential()
        # Conv2D(필터의 개수, 필터의 크기, 연산 시 필터가 이동하는 폭:strides, 활성함수, ..)
        model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=self.state_size))
        model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
        model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))

        # 플랫, relu 통과
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))

        # 출력은 행동의 개수만큼 노드의 개수를 가져야 함
        model.add(Dense(self.action_size))
        model.summary()
        return model

    # 타깃 모델을 모델의 가중치로 업데이트
    def update_target_model(self):  # 시작 전에 두 모델의 가중치 값을 통일해야 함
        self.target_model.set_weights(self.model.get_weights())

    # e-greeedy로 행동 선택
    def get_action(self, history):
        history = np.float32(history / 255.0)
        # 갈수록 무작위로 행동하는 확률이 적어짐
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model.pridict(history)
            return np.argmax(q_value[0])

    #샘플을 메모리에 저장함
    def append_sample(self, history, action, reward, next_history, dead):
        self.memory.append((history, action, reward, next_history, dead))

    # 리플레이 메모리에서 무작위로 추출한 배치로 모델 학습
    def train_model(self):
        if self.epsilon > self.epsilon_end:
            self.epsilon -= self.epsilon_decay_step

        # 메모리에서 배치 크기만큼 무작위로 샘플추출, mini batch로 학습
        mini_batch = random.sample(self.memory, self.batch_size)

        # 미리 형태를 지정함
        history = np.zeros((self.batch_size, self.state_size[0], self.state_size[1], self.state_size[2]))
        next_history = np.zeros((self.batch_size, self.state_size[0], self.state_size[1], self.state_size[2]))
        target = np.zeros((self.batch_size,))
        action, reward, dead = [], [], []

        # 무작위로 추출한 샘플을 집어 넣음
        for i in range(self.batch_size):
            # states[i] = mini_batch[i][0]
            # next_states[i] = mini_batch[i][3]
            history[i] = np.float32(mini_batch[i][0] / 255.)
            next_history[i] = np.float32(mini_batch[i][3] / 255.)

            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            dead.append(mini_batch[i][4])

        # 현재 상태에 대한 모델의 큐함수
        # target = self.model.predict(states)

        # 다음 상태에 대한 타깃 모델의 큐함수
        target_value = self.target_model.predict(next_history)

        #벨만 최적 방정식을 이용한 업데이트 타깃
        for i in range(self.batch_size):
            if dead[i]:    # 죽은 경우 Rt+1만 정답
                # target[i][actions[i]] = rewards[i]
                target[i] = reward[i]
            else:   # 정답 = Rt+1 + (gamma) * a'maxQ(St+1,a',(theta)-)
                target[i] = reward[i] + self.discount_factor * np.amax(target_value[i])

        # model.fit이 아니라 optimizer로 모델을 업데이트 : 오류함수로 MSE를 사용하지 않기 때문
        # self.model.fit(states, target, batch_size=self.batch_size, epochs=1, verbose=0)
        loss = self.optimizer([history, action, target])
        self.avg_loss += loss[0]

    # 각 에피소드 당 학습 정보를 기록
    def setup_summary(self):
        episode_total_reward = tf.Variable(0.)
        episode_avg_max_q = tf.Variable(0.)
        episode_duration = tf.Variable(0.)
        episode_avg_loss = tf.Variable(0.)

        tf.summary.scalar('Total Reward/Episode', episode_total_reward)
        tf.summary.scalar('Average Max Q/Episode', episode_avg_max_q)
        tf.summary.scalar('Duration/Episode', episode_duration)
        tf.summary.scalar('Average Loss/Episode', episode_avg_loss)

        summary_vars = [episode_total_reward, episode_avg_max_q, episode_duration, episode_avg_loss]
        summary_placeholders = [tf.placeholder(tf.float32) for _ in range(len(summary_vars))]
        update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in range(len(summary_vars))]
        summary_op = tf.summary.merge_all()
        return summary_placeholders, update_ops, summary_op

# 전처리 함수 : 학습 속도를 높이기 위해 흑백화면으로 전처리
def pre_processing(observe):
    # observe : 전처리 전 이미지, 하나의 화면, 사이즈는 [210, 160, 3]
    processed_observe = np.uint8(resize(rgb2gray(observe), (84, 84), mode='constant') * 255)
    # processed_observe : 전처리 후 이미지, 사이즈는 [84, 84, 1]
    return processed_observe

if __name__ == "__main__":
    env = gym.make('BreakoutDeterministic-v4')

    # state_size = env.observation_space.shape[0]
    # action_size = not env.action_space
    # agent = DQNAgent(state_size,action_size)
    agent = DQNAgent(action_size=3)

    scores, episodes, global_step = [], [], 0

    for e in range(EPISODES):
        done = False
        dead = False
        # score = 0
        step, score, start_life = 0, 0, 5
        # env 초기화
        # state = env.reset()
        # state = np.reshape(state, [1, state_size])
        observe = env.reset()

        # 브레이크 아웃 초반의 Agent의 오류를 방지하기 위해 일정 구간동안 아무것도 하지 않음
        for _ in range(random.randint(1, agent.no_op_steps)):
            observe, _, _, _ = env.step(1)

        state = pre_processing(observe)
        # 첫 history는 같은 state(전처리된 화면)을 4개 쌓음, state 사이즈가 [84, 84, 1]이기 때문에 1에 해당하는 축에 쌓음 -> (axis=2)
        history = np.stack((state, state, state, state), axis=2)
        history = np.reshape([history], (1, 84, 84, 4))
        # history 사이즈는 [84, 84, 4]

        while not done:
            if agent.render:
                env.render()
            global_step += 1
            step +=1

            # # 현재 상태로 행동을 선택
            # action = agent.get_action(state)
            # # 선택한 행동으로 환경에서 한 타임스텝 진행
            # history, reward, done, info = env.step(action)
            # next_state = np.reshape(next_state, [1, state_size])

            # 바로 전 4개의 상태로 행동을 선택
            action = agent.get_action(history)
            # 1 : 정지, 2 : 왼쪽, 3 : 오른쪽
            if action == 0:
                real_action = 1
            elif action == 1:
                real_action = 2
            else:
                real_action = 3

            # 선택한 행동으로 환경에서 한 타임스텝 진행
            observe, reward, done, info = env.step(real_action)

            # 각 타임스텝마다 상태 전처리
            next_state = pre_processing(observe)
            nest_state = np.reshape([next_state], (1, 84, 84, 1))
            # 오래된 state는 버리고 전처리를 거친 새로운 state를 채워주어야함
            next_history = np.append(next_state, history[:, :, :, :3], axis=3)

            agent.avg_q_max += np.amax(agent.model.predict(np.float32(history / 255.))[0])

            if start_life > info['ale.lives']:
                dead = True
                start_life = info['ale.lives']

            # 에피소드가 중간에 끝나면 -100 보상
            # reward = reward if not done or score == 499 else -100
            reward = np.clip(reward, -1., 1.)

            # 리플레이 메모리에 샘플 <s, a, r, s'> 저장
            agent.append_sample(history, action, reward, next_history, dead)
            # 매 타임스텝마다 학습
            if len(agent.memory) >= agent.train_start:
                agent.train_model()

            # 일정 시간(update_target_rate)마다 타깃 모델을 모델의 가중치로 업데이트
            if global_step % agent.update_target_rate == 0:
                agent.update_target_model()

            score += reward
            # state = next_state

            if dead:
                dead = False
            else:
                history = next_history

            if done:
                # # 각 에피소드마다 타깃 모델을 모델의 가중치로 업데이트
                # agent.update_target_model()
                # 각 에피소드마다 학습 정보를 기록
                if global_step > agent.train_start:
                    stats = [score, agent.avg_q_max / float(step), step, agent.avg_loss / float(step)]
                    for i in range(len(stats)):
                        agent.sess.run(agent.update_ops[i], feed_dict={
                            agent.summary_placeholders[i]: float(stats[i])
                        })
                    summary_str = agent.sess.run(agent.summary_op)
                    agent.summary_writer.add_summary(summary_str, e + 1)
                print("episode: ", e, " score: ", score, " memory length: ", len(agent.memory), " epsilon: ", agent.epsilon, " global_step: ", global_step, " average_q: ", agent.avg_q_max / float(step), " average loss: ", agent.avg_loss / float(step))

                agent.avg_q_max, agent.avg_loss = 0, 0

                # score = score if score == 500 else score + 100
                # # 에피소드마다 학습 결과 출력
                # scores.append(score)
                # episodes.append(e)
                # pylab.plot(episodes, scores, 'b')
                # pylab.savefig("./save_graph/cartpole_dqn.png")
                # print("episode:", e, "  score:", score, "  memory length:",
                #       len(agent.memory), "  epsilon:", agent.epsilon)
                #
                # # 이전 10개 에피소드의 점수 평균이 490보다 크면 학습 중단
                # if np.mean(scores[-min(10, len(scores)):]) > 490:
                #     agent.model.save_weights("./save_model/cartpole_dqn.h5")
                #     sys.exit()

            # 1000 에피소드마다 모델 저장
            if e % 1000 == 0:
                agent.model.save_weights("./save_model/breakout_dqn.h5")
