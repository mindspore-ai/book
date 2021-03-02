import gym
import math
import random
import numpy as np
import gym
from IPython import display
import matplotlib.pyplot as plt

import mindspore
import mindspore.nn as nn
from mindspore import context
from mindspore import Tensor
import mindspore.ops as ops


context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")


class DQN(nn. Cell):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        self.linear1 = nn.Dense(input_size, hidden_size)
        self.linear2 = nn.Dense(hidden_size, output_size)
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return x


class TARGET_DQN(nn. Cell):
    def __init__(self, input_size, hidden_size, output_size):
        super(TARGET_DQN, self).__init__()
        self.linear1 = nn.Dense(input_size, hidden_size)
        self.linear2 = nn.Dense(hidden_size, output_size)
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return x


class MyWithLossCell(nn.Cell):
    def __init__(self, backbone, loss_fn):
        super(MyWithLossCell, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._loss_fn = loss_fn
        self.gather = ops.GatherD()

    def construct(self, x, a0, label):
        out = self._backbone(x)
        out = self.gather(out, 1, a0)
        loss = self._loss_fn(out, label)
        return loss

    @ property
    def backbone_network(self):
        return self._backbone


class Agent(object):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.policy_net = DQN(self.state_space_dim, 256, self.action_space_dim)
        self.target_net = TARGET_DQN(self.state_space_dim, 256, self.action_space_dim)
        self.optimizer = nn.RMSProp(self.policy_net.trainable_params(), learning_rate=self.lr)
        loss_fn = nn.MSELoss()
        loss_Q_net = MyWithLossCell(self.policy_net, loss_fn)
        self.policy_net_train = nn.TrainOneStepCell(loss_Q_net, self.optimizer)
        self.policy_net_train.set_train(mode=True)
        self.buffer = []
        self.steps = 0

    def act(self, s0):
        self.steps += 1
        epsi = self.epsi_low + (self.epsi_high - self.epsi_low) * (math.exp(-1.0 * self.steps / self.decay))
        if random.random() < epsi:
            a0 = random.randrange(self.action_space_dim)
        else:
            s0 = np.expand_dims(s0, axis=0)
            s0 = Tensor(s0, mindspore.float32)
            a0 = self.policy_net(s0).asnumpy()
            a0 = np.argmax(a0)
        return a0

    def put(self, *transition):
        if len(self.buffer) == self.capacity:
            self.buffer.pop(0)
        self.buffer.append(transition)

    def load_dict(self):
        for target_item, source_item in zip(self.target_net.parameters_dict(), self.policy_net.parameters_dict()):
            target_param = self.target_net.parameters_dict()[target_item]
            source_param = self.policy_net.parameters_dict()[source_item]
            target_param.set_data(source_param.data)

    def learn(self):
        if (len(self.buffer)) < self.batch_size:
            return

        samples = random.sample(self.buffer, self.batch_size)
        s0, a0, r1, s1 = zip(*samples)
        s1 = Tensor(s1, mindspore.float32)
        s0 = Tensor(s0, mindspore.float32)
        a0 = Tensor(np.expand_dims(a0, axis=1))
        next_state_values = self.target_net(s1).asnumpy()
        next_state_values = np.max(next_state_values, axis=1)

        y_true = r1 + self.gamma * next_state_values
        y_true = Tensor(np.expand_dims(y_true, axis=1), mindspore.float32)
        self.policy_net_train(s0, a0, y_true)


def plot(score, mean):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.figure(figsize=(20, 10))
    plt.clf()

    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(score)
    plt.plot(mean)
    plt.text(len(score) - 1, score[-1], str(score[-1]))
    plt.text(len(mean) - 1, mean[-1], str(mean[-1]))


if __name__ == '__main__':
    env = gym.make('CartPole-v0')

    params = {
        'gamma': 0.8,
        'epsi_high': 0.9,
        'epsi_low': 0.05,
        'decay': 200,
        'lr': 0.001,
        'capacity': 100000,
        'batch_size': 512,
        'state_space_dim': env.observation_space.shape[0],
        'action_space_dim': env.action_space.n
    }
    agent = Agent(**params)

    score = []
    mean = []
    agent.load_dict()
    for episode in range(500):
        s0 = env.reset()
        total_reward = 1
        while True:
            a0 = agent.act(s0)
            s1, r1, done, _ = env.step(a0)

            if done:
                r1 = -1

            agent.put(s0, a0, r1, s1)

            if done:
                break

            total_reward += r1
            s0 = s1
            agent.learn()
        agent.load_dict()

        score.append(total_reward)
        mean.append(sum(score[-100:]) / 100)
        print("episode", episode, "mean", mean)
