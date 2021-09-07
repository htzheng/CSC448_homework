#!/usr/bin/python3

import numpy as np
import argparse

def load_weights(path):
    labels = []
    words = []
    with open(path, 'r') as f:
        for line in f.readlines():
            line = line.split('\n')[0]
            k, w = line.split(' ')
            l, a, b = k.split('_')
            if l == 'E':
                words.append(b)
            else:
                labels.append(a)
    labels = {k: i for i, k in enumerate(sorted(set(labels)))}
    words = {k: i for i, k in enumerate(sorted(set(words)))}

    emission = np.full((len(labels), len(words)), 0.0)
    transition = np.full((len(labels), len(labels)), 0.0)

    with open(path, 'r') as f:
        for line in f.readlines():
            line = line.split('\n')[0]
            k, w = line.split(' ')
            l, a, b = k.split('_')
            if l == 'E':
                emission[labels[a], words[b]] = float(w)
            elif l == 'T':
                transition[labels[a], labels[b]] = float(w)
            else:
                raise NotImplementedError
    return emission, transition, labels, words



def load_data(path):
    input_data = []
    gt_tag = []
    with open(path, 'r') as f:
        for line in f.readlines():
            line = line.split(' \n')[0]
            N = int(line.split(' ')[0])
            X = line.split(' ')[1::2]
            Y = line.split(' ')[2::2]
            X, Y = list(zip(*[[x, y] for x, y in zip(X, Y)]))
            input_data.append({'X': X, 'Y': Y, 'N': N})
            gt_tag.append(Y)
    return input_data, gt_tag



def decoding(emission, transition, label_to_idx, word_to_idx, data, verbose=True):
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    idx_to_word = {v: k for k, v in word_to_idx.items()}
    T = len(emission)
    results = []
    for sidx, sample in enumerate(data):

        N = sample["N"]
        cost = np.zeros((N, T))
        for i in range(N):
            word = sample["X"][i]
            if i == 0 and word in word_to_idx:
                x = word_to_idx[word]
                cost[i] = np.max(emission[:, None, x], axis=0)
            elif i == 0 and word not in word_to_idx:
                pass
            elif word in word_to_idx:
                x = word_to_idx[word]
                cost[i] = np.max(cost[i - 1,:,None] + emission[None,:,x] + transition, axis=0)
            else:
                cost[i] = np.max(cost[i - 1,:,None] + transition, axis=0)

        out = []
        max_score = np.max(cost[N-1])
        t = np.argmax(cost[N-1])
        out.append(t)
        for i in list(range(N-1))[::-1]:
            word = sample["X"][i]
            if word in word_to_idx:
                x = word_to_idx[word]
                t = np.argmax(emission[:,x] + transition[:,t])
            else:
                t = np.argmax(transition[:, t])
            out.append(t)

        result =  [idx_to_label[i] for i in out][::-1]
        if verbose:
            print(' '.join([str(max_score)]+result))
        results.append(result)
    return results

def evaluate(pred_tag, gt_tag):
    accuracy = 0.0
    for pred, gt in zip(pred_tag, gt_tag):
        correct_count = 0
        for a, b in zip(pred, gt):
            if a == b:
                correct_count += 1
        accuracy += float(correct_count)/len(pred)
    accuracy = accuracy / len(pred_tag)
    return accuracy

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Homework 1')
    parser.add_argument('weights', type=str)
    parser.add_argument('data', type=str)

    args = parser.parse_args()

    emission, transition, labels, words = load_weights(args.weights)
    input_data, gt_tag = load_data(args.data)

    pred_tag = decoding(emission, transition, labels, words, input_data)
    acc = evaluate(pred_tag, gt_tag)
    print('accuracy :{}'.format(acc))