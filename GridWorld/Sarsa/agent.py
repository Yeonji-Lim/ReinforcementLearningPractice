import numpy as np
import random
from collections import defaultdict
from environment import Env

class SARSAAgent:
    def __init__(self, actions):
        # action 정보는 환경으로부터 받아온다.
        self.actions = actions

        # learning_rate는 step size
        self.learning_rate = 0.01

        # 감가율
        self.discount_factor = 0.9

        # e-greedy policy
        self.epsilon = 0.1

        # defaultdict()은 dictionary에 기본값을 정의함
        # lambda를 통해 기본값을 설정할 수 있다.
        # 여기서는 큐함수를 의미한다.
        self.q_table = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])

    # e-greedy에 따라 행동 반환
    def get_action(self, state):
        # e-greedy
        if np.random.rand() < self.epsilon:
            # 무작위 행동 반환
            action = np.random.choice(self.actions)
        else:
            # 큐함수에 따른 행동 반환

            # 해당 상태에 해당하는 큐함수를 불러옴
            state_action = self.q_table[state]

            # 큐함수 중에서 최댓 값에 해당하는 인덱스를 행동으로 반환
            action = self.arg_max(state_action)

        return action

    # <s, a, r, s', a'>의 샘플로부터 큐함수 업데이트
    def learn(self, state, action, reward, next_state, next_action):
        current_q = self.q_table[state][action]
        next_state_q = self.q_table[next_state][next_action]

        # 살사의 큐함수 업데이트 식 : Q(s,a) = Q(s,a) + [step_size]*(R + [dicsount_factor]*Q(s',a') - Q(s,a))
        new_q = (current_q + self.learning_rate * (reward + self.discount_factor * next_state_q - current_q))
        self.q_table[state][action] = new_q

    #큐함수가 최대인 인덱스를 반환
    @staticmethod
    def arg_max(q_list):
        # 주석 처리된 거는 옛날 코드
        # max_index_list = []
        # max_value = state_action[0]
        # for index, value in enumerate(state_action):
        #     if value > max_value:
        #         max_index_list.clear()
        #         max_value = value
        #         max_index_list.append(index)
        #     elif value == max_value:
        #         max_index_list.append(index)
        # return random.choice(max_index_list)
        max_idx_list = np.argwhere(q_list == np.amax(q_list))
        max_idx_list = max_idx_list.flatten().tolist()
        return random.choice(max_idx_list)

if __name__ == "__main__":
    env = Env()
    agent = SARSAAgent(actions=list(range(env.n_actions)))

    for episode in range(1000):
        # 게임 환경과 상태를 초기화
        state = env.reset()
        # 현재 상태에 대한 행동 선택
        action = agent.get_action(str(state))

        while True:
            env.render()

            # 행동을 취한 후 환경에서 한 타임스텝 진행
            next_state, reward, done = env.step(action)

            # 다음 상태에서의 다음 행동
            next_action = agent.get_action(str(next_state))

            # <s,a,r,s',a'>로 큐함수를 업데이트
            agent.learn(str(state), action, reward, str(next_state), next_action)

            state = next_state
            action = next_action

            #모든 큐함수를 화면에 표시
            env.print_value_all(agent.q_table)

            if done:
                break
