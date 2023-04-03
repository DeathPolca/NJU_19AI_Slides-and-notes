# encoding=utf-8
import numpy as np


def fall_into_cliff(row, col):
    if row == 3:
        if col > 0 and col < 11:  # 37-46号，悬崖部分
            return 1
    else:
        return 0


def arrive(row, col):
    if row == 3 and col == 11:
        return 1
    else:
        return 0


class Env:

    def __init__(self, row, col):
        self.pos_row = row
        self.pos_col = col
        self.state = self.pos_row * 12 + self.pos_col

    def transition(self, action):
        if (action < 2):
            if (action == 0):
                self.pos_row = self.pos_row - 1 if self.pos_row > 0 else self.pos_row  # 0向上
            else:
                self.pos_row = self.pos_row + 1 if self.pos_row < 3 else self.pos_row  # 1向下
        else:
            if (action == 2):
                self.pos_col = self.pos_col - 1 if self.pos_col > 0 else self.pos_col  # 2向左
            else:
                self.pos_col = self.pos_col + 1 if self.pos_col < 11 else self.pos_col  # 3向右

        if (fall_into_cliff(self.pos_row, self.pos_col)):
            self.reset()
            return -100
        self.state = self.pos_row * 12 + self.pos_col

        if (arrive(self.pos_row, self.pos_col)):
            return 100

        return -1

    def reset(self):
        self.pos_row = 3
        self.pos_col = 0
        self.state = self.pos_row * 12 + self.pos_col


def epsilon_greedy(policy, state, epsilon):
    # 探索
    if np.random.uniform(0, 1) < epsilon:
        return np.random.randint(0, 4)  # 随机选一个方向走

    # 利用
    else:
        return policy[state]


def print_policy(i):
    if i == 0:
        print('^', end=" ")
    elif i == 1:
        print('v', end=' ')
    elif i == 2:
        print('<', end=' ')
    else:
        print('>', end=' ')


if __name__ == "__main__":
    e = 0.1
    alpha = 0.5
    gamma = 1

    # Sarsa
    # t = 100000  # 走的步数
    # agent = Env(3, 0)
    # q_table = np.zeros([48, 4])
    # policy = []
    # for i in range(0, 48):
    #     policy.append(np.random.randint(0, 4))  # 随机初始化策略,0上，1下，2左，3右
    #
    # state = agent.state
    # action = np.random.randint(0, 4)
    # for i in range(0, t):
    #     r = agent.transition(action)
    #     next_state = agent.state
    #     next_action = epsilon_greedy(policy, next_state, e)
    #     q_table[state, action] += alpha * (r + gamma * q_table[next_state, next_action] - q_table[state, action])
    #     policy[state] = np.argmax(q_table[state, :])
    #     action = next_action
    #     state = next_state

    # Q-learning
    t = 10000
    policy = []
    for i in range(0, 48):
        policy.append(np.random.randint(0, 4))  # 原始策略为随机策略，0上，1下，2左，3右，
    agent = Env(3, 0)  # 初始化状态
    state = agent.state
    q_table = np.zeros([48, 4])  # q值全部初始化为0
    for i in range(0, t):
        # print(agent.state,i,sep=' ')
        action = epsilon_greedy(policy, agent.state, e)
        r = agent.transition(action)
        next_state = agent.state
        next_action = policy[next_state]
        q_table[state, action] += alpha * (r + gamma * q_table[next_state, next_action] - q_table[state, action])
        policy[state] = np.argmax(q_table[state, :])
        action = next_action
        state = next_state

    # 输出策略
    # for i in range(0, 48):
    #     policy[i] = np.argmax(q_table[i, :])
    for i in range(0, 48):
        if i == 11 or i == 23 or i == 35:
            print_policy(policy[i])
            print('\n')
        else:
            print_policy(policy[i])
