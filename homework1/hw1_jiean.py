# -*- coding: utf-8 -*-
#
# Copyright © @pkuanjie <pkuanjie@gmail.com>.
#
# 2021-08-31 17:47
#
# Distributed under terms of the MIT license.

"""
This is the code for the hw1 of the csc448. This is an implementation of the Viterbi decoder algorithm.
"""

import sys
import os
import numpy as np
from copy import deepcopy

NINF = -1e8

def preprocess_weight(weights_raw):
    weights = dict()
    for line in weights_raw:
        name, value = line.strip().split(' ')
        weights[name] = float(value)
    return weights

def preprocess_data(data_raw, contain_label):
    data = []
    for line in data_raw:
        line = line.strip()
        line_list = line.split(' ')
        if contain_label:
            data_line_list = line_list[1: int(line_list[0]) * 2 + 1] # filter pure sentence records with word and pos pairs.
            data_line_list = [[data_line_list[i], data_line_list[i + 1]] for i in range(0, len(data_line_list) - 1, 2)]
        else:
            data_line_list = line_list[1: int(line_list[0]) + 1] # filter pure sentence records with word and pos pairs.

        data.append(data_line_list)

    return data

def get_T(weights):
    names = list(weights.keys())
    t_names = [i for i in names if i.startswith('T_')]
    e_names = [i for i in names if i.startswith('E_')]
    pos = []
    for item in t_names:
        _, a, b = item.split('_')
        pos.append(a)
        pos.append(b)

    for item in e_names:
        _, a, _ = item.split('_')
        pos.append(a)
    pos = list(set(pos))

    return pos

def read_weights_data(weights_path, data_path, contain_label):
    with open(weights_path, 'r') as f:
        weights_raw = f.readlines()
    with open(data_path, 'r') as f:
        data_raw = f.readlines()

    # preprocess weights and data to make it looks like:
    # weights: {E_xx_xx: value}
    # data: [[word, pos, word, pos, ...], [...], ...]
    weights = preprocess_weight(weights_raw)
    data = preprocess_data(data_raw, contain_label)
    pos = get_T(weights)
    return weights, data, pos

def get_item(data, i, contain_label):
    if contain_label:
        sentence = [item[0] for item in data[i]]
        gt = [item[1] for item in data[i]]
    else:
        sentence = data[i]
        gt = None
    return sentence, gt

def get_accuracy(path, gt):
    assert len(path) == len(gt)
    value = 0.0
    for i in range(len(path)):
        if path[i] == gt[i]:
            value += 1.0
    return value, len(path)

def psi(sentence, weights, names, pos_start, pos_end, i):
    t_name = 'T_%s_%s' % (pos_start, pos_end)
    e_name = 'E_%s_%s' % (pos_end, sentence[i])
    try:
        value_t = weights[t_name]
    except KeyError:
        value_t = 0.0

    try:
        value_e = weights[e_name]
    except KeyError:
        value_e = 0.0

    if i == 0:
        return value_e
    else:
        return value_t + value_e


def veterbi_decoder(weights, sentence, pos):
    #  print(sentence)
    N = len(sentence)
    T = len(pos)
    names = list(weights.keys())
    delta = np.zeros((N + 1, T))
    path_old = [[0]] * T
    path_new = [[0]] * T
    for i in range(1, N + 1):
        for t in range(T):
            #  print(pos[t])
            delta[i, t] = NINF
            for count, item in enumerate(pos):
                this_value = delta[i - 1, count] + psi(sentence, weights, names, item, pos[t], i - 1)
                #  print(psi(sentence, weights, names, item, pos[t], i - 1), this_value, sentence[i - 1], item, pos[t])
                if this_value >= delta[i, t]:
                    delta[i, t] = this_value
                    path_new[t] = path_old[count] + [pos[t]]
                    #  print(path_new[t])
        path_old = deepcopy(path_new)
    max_pos_index = np.argmax(delta[N, :])
    max_value = delta[N, max_pos_index]
    path = path_new[max_pos_index]
    return max_value, path[1:]

def main(weights_path, data_path, contain_label=True):
    weights, data, pos = read_weights_data(weights_path, data_path, contain_label)
    correct, all = 0.0, 0.0
    for i in range(len(data)):
        sentence, gt = get_item(data, i, contain_label)
        #  print(gt)
        max_value, path = veterbi_decoder(weights, sentence, pos)
        result = deepcopy(path)
        result.insert(0, max_value)
        print(' '.join(str(i) for i in result))
        if gt is not None:
            this_correct, this_all = get_accuracy(path, gt)
            correct += this_correct
            all += this_all
        exit()
    if contain_label:
        print('-------------------------------------')
        print("correct: %d, all: %d, accuracy: %.8f" % (correct, all, 1.0 * correct / all))

if __name__ == '__main__':
    weights_path = sys.argv[1]
    data_path = sys.argv[2]
    if len(sys.argv) == 4:
        contain_label = (sys.argv[3] == 'True')
    else:
        contain_label = False
    main(weights_path, data_path, contain_label)
