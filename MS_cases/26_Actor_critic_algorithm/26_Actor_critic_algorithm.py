
# 基于MindSpore实现AC算法

# gym强化学习库, gym版本为0.21.0
import gym
# 加载车杆游戏场景
env=gym.make('CartPole-v1')
# 获取状态数
state_number=env.observation_space.shape[0]
# 获取动作数
action_number=env.action_space.n

# 算法实现
## 导入Python库并配置运行信息
# os库
import os
# 引入MindSpore
import mindspore
# 引入神经网络模块
import mindspore.nn as nn
# 常见算子操作
import mindspore.ops as F
# 引入numpy
import numpy as np
# 引入张量模块
from mindspore import Tensor
# 配置静态库
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

## 定义参数和超参数

round_num = 10     # 训练回合数 
LR_A = 0.005        # learning rate for actor
LR_C = 0.01         # learning rate for critic
Gamma = 0.9         # 折现因子

## 定义损失函数

# 定义损失函数MAELoss
class MAELoss(nn.LossBase):
    def __init__(self, reduction="none"):
        super(MAELoss, self).__init__(reduction)
    # 构造损失函数
    def construct(self, prob, a,td):
        a_constant=a
        log_prob = F.log(prob)
        td_constant=td[0]
        # 定义信息熵
        log_prob_constant=-log_prob[0][a_constant]
        # 计算演员损失函数
        actor_loss = -log_prob_constant * td_constant
        return actor_loss

## 构建Actor网络


# Actor网络类
class ActorNet(nn.Cell):
    def __init__(self):
        super(ActorNet, self).__init__()
        # 全连接层1
        self.fc1 = nn.Dense(state_number, 50)
        # 全连接层2
        self.fc2 = nn.Dense(50, 20)
        # 全连接层3
        self.fc3 = nn.Dense(20, action_number)
        # ReLU激活函数
        self.relu = nn.ReLU()
        # Sigmoid激活函数
        self.sigmoid = nn.Sigmoid()
        # Softmax归一化函数
        self.softmax = nn.Softmax()
    # 构造Actor网络
    def construct(self, x):
        # 全连接层
        x = self.fc1(x)
        # ReLU激活函数
        x = self.relu(x)
        # 全连接层
        x = self.fc2(x)
        # Sigmoid激活函数
        x = self.sigmoid(x)
        # 全连接层
        x = self.fc3(x)
        # 返回softmax函数结果
        return self.softmax(x)
# 演员类
class Actor():
    def __init__(self):
        # 构造演员网络
        self.actor=ActorNet()
        # 优化器
        self.optimizer = nn.Adam(self.actor.trainable_params(),learning_rate=LR_A)
        # 损失函数
        self.loss=MAELoss()
    # 行为选择函数
    def choose(self,inputstate):
        # inputstate=Tensor(inputstate)
        # 输入状态
        inputstate=Tensor([inputstate])
        probs=self.actor(inputstate).asnumpy()
        # 获取行为
        action=np.random.choice(np.arange(action_number),p=probs[0])
        # 返回行为
        return action
    # 学习函数
    def learn(self,s,a,td):
        s = Tensor([s])
        prob = self.actor(s)
        # log_prob = F.log(prob)
        # td张量化
        td=Tensor(td)
        # a转为整型变量
        a=int(a)
        a_constant = a
        # 信息熵
        log_prob = F.log(prob)
        td_constant = td[0]
        log_prob_constant = -log_prob[0][a_constant]
        # 构造损失网络
        loss_net=nn.WithLossCell(self.actor,loss_fn=self.loss(prob,Tensor(a),td))
        # 构造训练网络
        train_net = nn.TrainOneStepCell(loss_net,self.optimizer)

## 构建Critic网络

# 评论者网络
class CriticNet(nn.Cell):
    def __init__(self):
        super(CriticNet, self).__init__()
        # 全连接层1
        self.fc1 = nn.Dense(state_number, 50)
        # 全连接层2
        self.fc2 = nn.Dense(50, 20)
        # 全连接层3
        self.fc3 = nn.Dense(20, 1)
        # ReLU激活函数
        self.relu = nn.ReLU()
        # Sigmoid激活函数
        self.sigmoid = nn.Sigmoid()
        # Softmax归一化函数
        self.softmax = nn.Softmax()
    # 构造函数
    def construct(self, x):
        # 全连接层
        x = self.fc1(x)
        # ReLU激活函数
        x = self.relu(x)
        # 全连接层
        x = self.fc2(x)
        # Sigmoid激活函数
        x = self.sigmoid(x)
        # 返回全连接层结果
        return self.fc3(x)
class Critic():
    def __init__(self):
        self.critic=CriticNet()
        self.tempA=ActorNet()
        # 优化器
        self.optimizer = nn.Adam(self.critic.trainable_params(),learning_rate=LR_C)
        # 均方误差（MSE）
        self.lossfunc=nn.MSELoss()
    def learn(self,s,r,s_):
        s = Tensor([s])
        # 输入当前状态，由网络得到估计v
        v=self.critic(s)
        r=Tensor([r])
        s_ = Tensor([s_])
        temp=Tensor(self.critic(s_).asnumpy())
        # 真实v
        reality_v=r+Gamma*temp[0]
        # 构造损失网络
        loss_net=nn.WithLossCell(self.critic,self.lossfunc(reality_v,v[0]))
        # 构造训练网络
        train_net=nn.TrainOneStepCell(loss_net,self.optimizer)
        # 计算真实v与估计v之间的差距
        advantage=(reality_v-v).asnumpy()
        return advantage

# 模型训练

print('AC训练中...')
# 演员实例化
a=Actor()
# 评论者实例化
c=Critic()
# 开始训练
for i in range(round_num):
    r_totle=[]
    # 环境重置
    observation = env.reset()#环境重置
    # 判断数据类型（gym版本）
    observation = observation if type(observation)== np.ndarray else observation[0]
    # print(observation)
    # print(observation)
    while True:
        # 选择行为
        action=a.choose(observation)
        # print(env.step(action))
        step = env.step(action)
        # 获取训练过程相关参数
        observation_ = step[0]
        reward = step[1]
        done = step[2]
        info = step[3]
        # 打印出训练过程
        # print(observation_, reward, done, info)
        # observation_, reward, done, info = env.step(action)
        # 获取td误差
        td_error =c.learn(observation,reward,observation_)
        # 学习行为
        a.learn(observation,action,td_error)
        observation=observation_
        # 行为选择奖励加入总回报
        r_totle.append(reward)
        # done = True进行下一轮训练
        if done:
            break
    # 计算总回报
    r_sum=sum(r_totle)
    # 打印训练回合数和奖励
    print("\r回合数：{} 奖励：{}".format(i,r_sum),end=" ")
    print(observation)
    # 保存检查点
    if i%100==0 and i > int(round_num/2):
        mindspore.save_checkpoint(a.actor, "./actor.ckpt")
        mindspore.save_checkpoint(c.critic, "./critic.ckpt")
print('AC训练完成')
exit(0)
