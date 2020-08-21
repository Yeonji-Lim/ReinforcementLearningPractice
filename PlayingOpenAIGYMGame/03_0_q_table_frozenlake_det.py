# https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0#.pjz9g59ap

import gym
import numpy as np
import matplotlib.pyplot as plt
from gym.envs.registration import register
import random as pr

# https://gist.github.com/stober/1943451


def rargmax(vector):
    """ Argmax that chooses randomly among eligible maximum indices. """
    m = np.amax(vector)
    indices = np.nonzero(vector == m)[0]
    return pr.choice(indices)


register(
    id='FrozenLake-v3',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name': '4x4',
            'is_slippery': False}
)
env = gym.make('FrozenLake-v3')

# Initialize table with all zeros
Q = np.zeros([env.observation_space.n, env.action_space.n])
# 테이블 이름이 Q  ^상태의 개수 : 16           ^액션의 개수 : 4

# Set learning parameters
num_episodes = 2000
# 몇 번의 학습을 시킬 지

# create lists to contain total rewards and steps per episode
rList = []

for i in range(num_episodes):
    # Reset environment and get first new observation
    state = env.reset()
    rAll = 0
    done = False # 게임이 끝났는 지 여부

    # The Q-Table learning algorithm
    while not done:
        action = rargmax(Q[state, :])
        # Q 중에서 가장 큰 값을 가져온다. 그런데 같다면 랜덤하게 가져옴 예를 들어 처음에 다 0일때 랜덤

        # Get new state and reward from environment
        new_state, reward, done, _ = env.step(action)

        # Update Q-Table with new knowledge using learning rate
        Q[state, action] = reward + np.max(Q[new_state, :])

        # 반환값
        rAll += reward

        # 상태 업데이트
        state = new_state

    # 해당 에피소드의 반환값을 저장한다.
    rList.append(rAll)


print("Success rate: " + str(sum(rList) / num_episodes))
print("Final Q-Table Values")
print("LEFT DOWN RIGHT UP")
print(Q)

# 그래프 그리는 부분
plt.bar(range(len(rList)), rList, color="blue")
plt.show()
