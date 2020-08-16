import gym
from gym.envs.registration import register

import readchar  # pip3 install readchar

# MACROS
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

# Key mapping
arrow_keys = {
    '\x1b[A': UP,
    '\x1b[B': DOWN,
    '\x1b[C': RIGHT,
    '\x1b[D': LEFT}

# Register FrozenLake with is_slippery False
register(
    id='FrozenLake-v3',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name': '4x4', 'is_slippery': False}
)
# 환경을 미끄럽지 않게 설정 : 행동외에 다른 변수가 없음

env = gym.make('FrozenLake-v3') #환경 설정
env.render()  # Show the initial board

while True:
    # Choose an action from keyboard
    key = readchar.readkey()
    # 이상한 키 누른 경우
    if key not in arrow_keys.keys():
        print("Game aborted!")
        break

    # action 선정
    action = arrow_keys[key]
    # 환경에서 action 수행 및 저장
    state, reward, done, info = env.step(action)
    env.render()  # Show the board after action
    print("State: ", state, "Action: ", action,
          "Reward: ", reward, "Info: ", info)

    if done:
        print("Finished with reward", reward)
        break