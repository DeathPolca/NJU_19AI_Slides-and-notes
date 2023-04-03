import os
import numpy as np
from scipy.optimize import linprog
from scipy.spatial import HalfspaceIntersection as HSI
from itertools import product
import time


def pole_find(B, b):
    poles = []
    temp = []
    for i in range(len(B)):
        sum = 0
        for j in range(len(B[0])):
            sum += B[i][j] ** 2
        temp.append([np.sqrt(sum)])
    A = np.concatenate((B, temp), axis=1)
    c = np.array([0] * (A.shape[1] - 1) + [-1])
    '''求可行点'''
    res = linprog(c=c, A_ub=A, b_ub=b)
    best_res = res.x[:-1]
    ndim_plus1 = []
    for i in b:
        ndim_plus1.append([-i])
    '''HSI第一个参数表示Ax+b<0'''
    hs = HSI(np.concatenate((B, ndim_plus1), axis=1), best_res)
    hs.close()
    '''flag1判定是否存在无限大的元素(不能存在)，flag2判定是否存在非0元素(有1个以上就可以)'''
    for v in hs.intersections:
        flag1 = 1
        flag2 = 0
        for item in v:
            if not np.isfinite(item):
                flag1 = 0
            if np.abs(item) > 1e-8:
                flag2 = 1
        if flag1 == 1 and flag2 == 1:
            label = np.where(np.abs(B.dot(v) - b) < 1e-6)[0]
            poles.append((v, label))
    return poles


def get_MNE(actions, payoffs):
    NEs = set()
    payoff_P1 = payoffs[0]  # P1收益矩阵
    payoff_P2 = payoffs[1]  # P2收益矩阵
    '''算法要求收益矩阵为非负的'''
    for i in payoffs[0].flatten():
        if i < 0:
            payoff_P1 += np.abs(np.min(payoffs[0]))
            break
    for i in payoffs[1].flatten():
        if i < 0:
            payoff_P2 += np.abs(np.min(payoffs[1]))
            break

    '''P1的优化问题参数，要把x>0和B^Tx<1写到一起'''
    A = np.concatenate((payoff_P1, -np.identity(payoffs[0].shape[1])))  # 负单位矩阵保证x>=0
    a = np.array([1] * payoffs[0].shape[0] + [0] * payoffs[0].shape[1])
    '''P2的优化问题参数，与P1的同理'''
    B = np.concatenate((payoff_P2.T, -np.identity(payoffs[1].shape[0])))
    b = np.array([1] * payoffs[0].shape[1] + [0] * payoffs[0].shape[0])
    nes, label = set(), set(range(sum(actions)))
    pole1 = pole_find(A, a)
    pole2 = pole_find(B, b)
    for i in range(len(pole1)):
        label1 = pole1[i][1]
        for j in range(len(pole2)):
            label2 = (pole2[j][1] + actions[0]) % sum(actions)
            if set(list(label1) + list(label2)) == label:
                ne = list(pole2[j][0] / np.sum(pole2[j][0])) + list(pole1[i][0] / np.sum(pole1[i][0]))
                NEs.add(tuple(ne))
    return NEs


def get_PNE(n_players, actions, payoffs):
    all_best_res = dict()  # 记录每个玩家的最优反应集合
    for i in range(n_players):
        payoff = payoffs[i]  # 第i个玩家的奖励
        best_res = set()  # 记录其对不同对手行为的最优反应
        '''先给出对第i个玩家来说，其对手的所有行为组合all_action_set，其中i的行为被设置为none'''
        all_action = []
        for j in range(n_players):
            if j == i:
                all_action.append([None])
            else:
                all_action.append(range(actions[j]))

        all_action_set = product(*(all_action))
        '''接下来计算最优反应'''
        for strategy in all_action_set:
            game_res = [payoff[(*strategy[0:i], j, *strategy[i + 1:])] for j in range(actions[i])]
            for j in np.argwhere(np.array(game_res) == max(game_res)):
                best_res.add((*strategy[0:i], j[0], *strategy[i + 1:]))
        all_best_res[i] = best_res
    '''find PNE'''
    PNE = set()
    '''取所有玩家最优反应的交集'''
    PNE_not_formalized = all_best_res[0]
    for i in range(1, n_players):
        PNE_not_formalized = PNE_not_formalized & all_best_res[i]
    '''改写成标准形式，即对每个玩家i都长为action[i]，且是onehot向量'''
    for p in PNE_not_formalized:
        pne = []
        for k in range(n_players):
            temp = [0] * actions[k]
            temp[p[k]] = 1
            pne = pne + temp
        PNE.add(tuple(pne))
    return PNE


def load_data(filename):
    with open(filename, 'r') as f:
        flag = 0
        payoffs = []
        tmp_payoff = []
        for line in f:
            if line[0:8] == "Players:":
                n_players = line[9]
            if line[0:8] == "Actions:":
                actions = list(line[9:-1].split(" "))
            if line.count('\n') == len(line):  # 意味着这一行是空行，后面全是收益
                flag = 1
                continue
            if flag == 1:
                tmp_payoff.append(line.split(" "))
    tmp_payoff[0].pop()
    for i in range(len(tmp_payoff[0])):
        tmp_payoff[0][i] = int(tmp_payoff[0][i])
    tmp_payoff = list(np.ravel(tmp_payoff))  # 把所有收益列成了一个长列表
    '''str转int'''
    n_players = int(n_players)
    for i in range(len(actions)):
        actions[i] = int(actions[i])
    for i in range(n_players):  # 一个玩家一个ndarray,n个玩家一组受益
        p = [tmp_payoff[j] for j in range(len(tmp_payoff)) if j % n_players == i]
        tmp = np.array([p])
        tmp = np.reshape(tmp, actions, order='F')
        payoffs.append(tmp)

    return n_players, actions, payoffs


def write_ne(result, out_path):
    with open(out_path, 'w') as f:
        for res in result:  # res是一个解,result是解得集合
            to_str = [str(i) if i > 1e-15 else '0' for i in res]
            tmp_str = to_str[0]
            for i in range(1, len(to_str) - 1):
                tmp_str = tmp_str + ',' + to_str[i]
            tmp_str = tmp_str + ',' + to_str[-1] + '\n'
            f.writelines(tmp_str)


def nash(in_path, out_path):
    # load file
    n_players, actions, payoffs = load_data(in_path)
    # get NE
    # 两人以上只找纯策略NE
    if n_players > 2:
        result = get_PNE(n_players, actions, payoffs)
    # 两人则找混合
    else:
        result = get_MNE(actions, payoffs)
    # write file
    write_ne(result, out_path)


if __name__ == '__main__':
    for f in os.listdir('input'):
        if f.endswith('.nfg'):
            t1 = time.time()
            nash('input/' + f, 'output/' + f.replace('nfg', 'ne'))
            t2 = time.time()
            print("computing " + f + " time costs: ", t2 - t1)
