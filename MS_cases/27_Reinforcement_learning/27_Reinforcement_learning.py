
# 基于MindSpore实现强化学习示例

# Hyper Parameters
BATCH_SIZE = 8
LR = 0.01                   # learning rate
EPSILON = 0.9               # greedy policy
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 100   # target update frequency
MEMORY_CAPACITY = 2900

### 数据加载

# 科学计算库
import numpy as np
import gym
# MindSpore库
import mindspore as ms
import mindspore.nn as nn
# 常见算子操作
from mindspore import ops

# 加载车杆模型
env = gym.make('CartPole-v1')
env = env.unwrapped
# 获取动作数
N_ACTIONS = env.action_space.n
# 获取状态数
N_STATES = env.observation_space.shape[0]
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape     # to confirm the shape

## 模型构建

### 导入Python库和模块

# 科学计算库
import numpy as np
import gym
# MindSpore库
import mindspore as ms
import mindspore.nn as nn
# 常见算子操作
from mindspore import ops
# 引入张量模块
from mindspore import Tensor

### 定义神经网络模型 Net

class Net(nn.Cell):
    def __init__(self, ):
        super(Net, self).__init__()
        # 一个全连接层
        self.fc1 = nn.Dense(N_STATES, 20)
        # 第二个全连接层
        self.fc2 = nn.Dense(20, 50)
        # 第s三个全连接层
        self.fc3 = nn.Dense(50, N_ACTIONS)
        # 激活函数
        self.relu = nn.ReLU()
        
    def construct(self, x):
        if x.ndim < 2:
            x = ops.expand_dims(x, 0)

        # 全连接层
        x = self.fc1(x)
        # 全连接层
        x = self.fc2(x)
        # 激活层
        x = self.relu(x)
        # 全连接层
        x = self.fc3(x)
        # 输出
        actions_value = self.relu(x)
        # 返回输出值
        return actions_value

### 构建DQN

#### 初始化（__init__）：
#### 选择动作（choose_action）：
#### 存储转换（store_transition）：
#### 学习过程（learn）：

class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()
        # 更新目标
        self.learn_step_counter = 0                                     # for target updating
        # 存储记忆
        self.memory_counter = 0                                         # for storing memory
        # 记忆初始化
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))     # initialize memory
        # Adaptive Moment Estimation 动态调整参数学习率
        self.optimizer = nn.Adam(self.eval_net.trainable_params(), learning_rate=LR)
        # MSE损失函数
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        # 输入样本
        x = Tensor([x])

        # greedy贪心
        if np.random.uniform() < EPSILON:   
            # 获取所有动作值
            actions_value = self.eval_net(x)
            # 获取最优动作
            action = np.array(ops.max(actions_value), dtype=int)

            # 调整维度
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)  # return the argmax index
            if type(action) == np.ndarray: action = action[0]
        else:   
            # 随机获取动作值
            action = np.random.randint(0, N_ACTIONS)
            # 调整维度
            action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
           
        action = np.array(action, dtype=int)

        # 返回动作值
        return action

    def store_transition(self, s, a, r, s_):

        transition = np.hstack((s, [a, r], s_))
        # 更新记忆内容
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # 目标参数更新
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            # 加载网络参数
            ms.load_param_into_net(self.target_net, self.eval_net.parameters_dict())
        self.learn_step_counter += 1
        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = Tensor([b_memory[:, :N_STATES]])
        b_a = Tensor([b_memory[:, N_STATES:N_STATES+1].astype(int)])
        b_r = Tensor(b_memory[:, N_STATES+1:N_STATES+2])
        b_s_ = Tensor([b_memory[:, -N_STATES:]])
        # 计算现实Q值
        q_eval = self.eval_net(b_s).gather_elements(2, b_a)  # shape (batch, 1
        # 从计算图中分离，阻止反向传播
        q_next = ops.diag(self.target_net(b_s_))
        # 保持张量数据类型一致
        b_r = b_r.astype('float32')
        # 计算目标Q值
        q_target = b_r + GAMMA * q_next[0][0][0].max(2).view((BATCH_SIZE, 1))   # shape (batch, 1)
        # 损失函数
        loss_net = nn.WithLossCell(self.eval_net, loss_fn = self.loss_func(q_eval, q_target))
        # 梯度更新
        train_net = nn.TrainOneStepCell(loss_net, self.optimizer)

## 模型训练

dqn = DQN()
print('开始训练...')
for i_episode in range(300):
    # 重置环境
    s = env.reset()
    s = s[0]    # get array element

    # 总奖励
    ep_r = 0
    while True:
        a = dqn.choose_action(s)

        if a.ndim > 0:
            a = a[0]

        # 获取训练过程相关参数
        # print(env.step(a))
        s_, r, done, info, _ = env.step(a)

        # 修改奖励
        x, x_dot, theta, theta_dot = s_
        # 计算奖励值
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        r = r1 + r2

        # 更新记忆
        dqn.store_transition(s, a, r, s_)

        # 计算总奖励
        ep_r += r

        if dqn.memory_counter > MEMORY_CAPACITY:
            # 从记忆中学习
            dqn.learn()
            if done:
                print('Epoch: ', i_episode, '| Epoch_reward: ', round(ep_r, 2))

        if done:
            break
        s = s_

exit(0)