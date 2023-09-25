
# 基于MindSpore实现Q-Learning算法的示例

## 实验准备
N_STATES = 6                      # 定义的环境下，有6个状态
ACTIONS = ["left", "right"]       # 有两个动作
EPSILON = 0.9                     # ε-greedy算法中的探索率
ALPHA = 0.1                       # 学习率
GAMMA = 0.9                       # 折扣因子
MAX_EPISODES = 15                 # 最大训练回合数
FRESH_TIME = 0.3                  # 刷新环境的时间间隔
TerminalFlag = "terminal"         # 有一个终止状态

## 数据加载

# gym强化学习库, gym版本为0.26.2
import gym
# 加载车杆游戏场景
env=gym.make('CartPole-v1')
# 获取状态数
state_number=env.observation_space.shape[0]
# 获取动作数
action_number=env.action_space.n

# 算法实现

## 导入Python库并配置运行信息
import time
import numpy as np
import pandas as pd
import mindspore as ms

## 设置实验环境和参数
# 设置计算环境
ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")

N_STATES = 6                      # 定义的环境下，有6个状态
ACTIONS = ["left", "right"]       # 有两个动作
EPSILON = 0.9                     # ε-greedy算法中的探索率
ALPHA = 0.1                       # 学习率
GAMMA = 0.9                       # 折扣因子
MAX_EPISODES = 15                 # 最大训练回合数
FRESH_TIME = 0.3                  # 刷新环境的时间间隔
TerminalFlag = "terminal"         # 有一个终止状态

## 建立Q表

def build_q_table(n_states, actions):
    return pd.DataFrame(
        np.zeros((n_states, len(actions))),
        columns=actions
    )

## 5.4 根据当前状态和Q表选择动作

def choose_action(state, q_table):
    state_table = q_table.loc[state, :]
    if (np.random.uniform() > EPSILON) or ((state_table == 0).all()):
        action_name = np.random.choice(ACTIONS)  # 以一定概率随机选择动作
    else:
        action_name = state_table.idxmax()       # 选择Q值最大的动作
    return action_name

## 获取环境反馈

def get_env_feedback(S, A):
    if A == "right":
        if S == N_STATES - 2:
            S_, R = TerminalFlag, 1  # 到达终止状态的奖励为1
        else:
            S_, R = S + 1, 0         # 向右移动的奖励为0
    else:
        S_, R = max(0, S - 1), 0    # 向左移动的奖励为0
    return S_, R

## 更新环境


# 用于更新环境，展示当前状态和可选的动作
def update_env(S, episode, step_counter): 
    env_list = ["-"] * (N_STATES - 1) + ["T"]
    if S == TerminalFlag:
        interaction = 'Episode %s: total_steps = %s' % (episode + 1, step_counter)
        print(interaction)
        time.sleep(2)
    else:
        env_list[S] = '0'
        interaction = ''.join(env_list)
        print(interaction)
        time.sleep(FRESH_TIME)


# 模型训练

def rl():
    q_table = build_q_table(N_STATES, ACTIONS)  # 建立初始的 Q 表
    for episode in range(MAX_EPISODES):  # 进行多个回合的训练
        step_counter = 0  # 记录每个回合的步数
        S = 0  # 初始状态为 0
        is_terminated = False  # 判断是否终止的标志位
        update_env(S, episode, step_counter)  # 更新环境显示
        while not is_terminated:
            A = choose_action(S, q_table)  # 根据当前状态和 Q 表选择动作
            S_, R = get_env_feedback(S, A)  # 获取环境反馈，得到下一个状态和奖励值
            q_predict = q_table.loc[S, A]  # 获取当前状态和选择的动作的 Q 值

            if S_ != TerminalFlag:  # 如果下一个状态不是终止状态
                q_target = R + GAMMA * q_table.loc[S_, :].max()  # 计算目标 Q 值，采用贝尔曼方程更新
            else:
                q_target = R  # 如果下一个状态是终止状态，则目标 Q 值为当前的奖励值，没有后续的动作选择
                is_terminated = True  # 设置终止标志位为 True
            q_table.loc[S, A] += ALPHA * (q_target - q_predict)  # 更新 Q 表
            S = S_  # 更新当前状态为下一个状态
            update_env(S, episode, step_counter + 1)  # 更新环境显示
            step_counter += 1  # 步数加1
    return q_table

# 模型预测

if __name__ == '__main__':
    q_table = rl()
    print(q_table)
    exit(0)
