import sys
import os
from collections import OrderedDict
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm
import time

NINF = -1e9
MAX_LENGTH = 20
END='.'
END_TAG='.'
END_IDX = 0
START = '->'
START_TAG = '->'
START_IDX = 1
OOV='???OUT_OF_VOC???'
OOV_TAG='#???OUT_OF_VOC???'
OOV_IDX = 2

LR = 0.002
MAX_ITER = 100000
BATCH_SIZE = 500
PRINT_FREQ = 1000
L1_REG = 0.01

def load_weight(fname, tags, words, tags_idx, words_idx, T, W):
    weights = dict()
    with open(fname, 'r') as f:
        for line in f:
            key, value = line.strip().split(' ')
            weights[key] = float(value)

    Trans = NINF * torch.ones((T, T), dtype=torch.float32)
    Emis = NINF * torch.ones((T, W), dtype=torch.float32)

    for item in weights.keys():
        if item.startswith('T_'):
            _, a, b = item.split('_')
            a, b = a.lower(), b.lower()
            Trans[tags_idx[a], tags_idx[b]] = weights[item]
        elif item.startswith('E_'):    
            _, a, w = item.split('_')
            a, w = a.lower(), w.lower()
            Emis[tags_idx[a], words_idx[w]] = weights[item]

    return weights

def load_data(fname, add_start=True, add_end=True, padding=True, max_length=10):
    y = []
    x = []
    with open(fname, 'r') as f:
        for line in f:
            data = line.strip().split(' ')[1::]
            data = [START, START_TAG] + data if add_start else data
            data = data + [END, END_TAG] if add_end else data
            if padding and add_end: # pad end symbol multiple times
                data = data + [END, END_TAG] * (max_length-len(data)//2)
            data = data[0:2*max_length] # truncation
            assert len(data)%2==0
            x.append(data[0::2])
            y.append(data[1::2])
    return x, y

def to_index(data, index_table, exception_value):
    N, L = len(data), len(data[0])
    m = torch.zeros((N, L), dtype=torch.int64)
    for i in range(N):
        for j in range(L):
            m[i,j] = index_table.get(data[i][j], exception_value)
    return m

def sampler(x, y, batchsize=5, shuffle=True):
    N = y.size(0)
    l = list(range(N))
    if shuffle:
        while True:
            idx = random.sample(l, batchsize)
            yield x[idx,:], y[idx,:]
    raise NotImplementedError('none shuffle version not implemented')

''' decode batched '''
def get_tag_batched(pre, i, t):
    if i>0:
        return torch.cat((get_tag_batched(pre, i-1, torch.gather(pre[:,i], 1, t.unsqueeze(1)).squeeze(1)), t.unsqueeze(1)), dim=1)
    else:
        return torch.full_like(t, START_IDX, dtype=torch.int64).unsqueeze(1)

def decode_batch(x, y, trans, emis, device='cpu'):
    B, N = x.size()
    T, W = emis.size()

    ''' decode batched '''
    delta = torch.full((B, N, T), NINF, dtype=torch.float32, device=device)
    delta[:,0,START_IDX].fill_(0)
    pre = torch.zeros((B, N, T), dtype=torch.int64, device=device)
    for i in range(1, N):
        x_i = x[:,i]
        ''' faster version'''
        values = delta[:,i-1,:].unsqueeze(-1) + trans[:,:].unsqueeze(0) + emis[:,x_i].transpose(0,1).unsqueeze(1)
        delta[:,i,:], pre[:,i,:] = torch.max(values,1)

    _, max_idx = torch.max(delta[:,N-1, :], 1)
    decode_idx = get_tag_batched(pre, N-1, max_idx)
    accuracy = (decode_idx==y).float().mean().item()
    return decode_idx, accuracy

def forward_backward_log_batch(x, y, trans, emis, device='cpu'):
    B, N = x.size()
    T, W = emis.size()

    trans_logp = F.log_softmax(trans, 1)
    emis_logp = F.log_softmax(emis, 1)
    ''' forward batched '''
    log_a = torch.zeros((B, N, T), dtype=torch.float32, device=device)
    log_a[:,0,:].fill_(NINF)
    for i in range(1, N):
        log_a[:,i,:] = (emis_logp[:,x[:,i]].transpose(0,1).unsqueeze(1) + log_a[:,i-1,:].unsqueeze(2) + trans_logp.unsqueeze(0)).logsumexp(1)

    ''' backward batched '''
    log_b = torch.zeros((B, N, T), dtype=torch.float32, device=device)
    log_b[:,N-1,:] = - log_a[:,N-1,:].logsumexp(1).unsqueeze(1).expand_as(log_b[:,N-1,:])
    for i in range(N-2,-1,-1):
        log_b[:,i,:] = (emis_logp[:,x[:,i+1]].transpose(0,1).unsqueeze(-1) + trans_logp.transpose(0,1).unsqueeze(0) + log_b[:,i+1,:].unsqueeze(-1)).logsumexp(1)
    ''' marginal '''
    log_marginal = log_a+log_b

    _, decode_idx = torch.max(log_marginal, 2)
    accuracy = (decode_idx==y).float().mean().item()
    return log_marginal, decode_idx, accuracy

def word_count(train_x, train_y, cache='weight_count.pkl'):
    if os.path.isfile(cache):
        print('loading cache {cache}')
        with open(cache, 'rb') as f:
            trans = np.load(f)
            emis = np.load(f)
    else:
        trans = np.zeros((T, T), dtype=np.float32)
        emis = np.zeros((T, W), dtype=np.float32)
        for i in tqdm(range(train_x.size(0))):
            x, y = train_x[i], train_y[i]
            for j in range(x.size(0)):
                emis[y[j], x[j]] += 1
                if j<x.size(0)-1:
                    trans[y[j], y[j+1]] += 1
        with open(cache, 'wb') as f:
            np.save(f, trans)
            np.save(f, emis)
    # normalize
    trans += 1
    emis += 1
    trans = trans/trans.sum(1, keepdims=True)
    emis = emis/emis.sum(1, keepdims=True)
    # set for start symbol
    trans = torch.from_numpy(trans)
    emis = torch.from_numpy(emis)
    trans = torch.log(trans)
    emis = torch.log(emis)

    trans[:,START_IDX] = NINF
    emis[START_IDX,:] = NINF
    return trans, emis


if __name__ == '__main__':
    train_x_, train_y_ = load_data(sys.argv[1])
    test_x_, test_y_ = load_data(sys.argv[2])
    use_gpu = int(sys.argv[3])
    device = "cuda:0" if use_gpu else "cpu"

    ## create tags/words tables
    words = OrderedDict([(END, 0), (START, 0), (OOV, 0)])
    tags = OrderedDict([(END_TAG, 0), (START_TAG, 0)])
    for x, y in zip(train_x_, train_y_):
        for x_i, y_i in zip(x, y):
            words.update({x_i:0})
            tags.update({y_i:0})
    # for x, y in zip(test_x_, test_y_):
    #     for x_i, y_i in zip(x, y):
    #         words.update({x_i:0})
    #         tags.update({y_i:0})
    tags, words = list(tags), list(words)
    T, W, N = len(tags), len(words), MAX_LENGTH
    tags_idx = {name:idx for (idx,name) in enumerate(tags)}
    words_idx = {name:idx for (idx,name) in enumerate(words)}
    
    ## convert tags/words to index
    train_x, train_y = to_index(train_x_, words_idx, OOV_IDX).to(device), to_index(train_y_, tags_idx, None).to(device)
    test_x, test_y = to_index(test_x_, words_idx, OOV_IDX).to(device), to_index(test_y_, tags_idx, None).to(device)

    ## weight by word count
    trans, emis = word_count(train_x, train_y)
    trans, emis = trans.to(device), emis.to(device)
    decode_idx, decode_accuracy = decode_batch(test_x, test_y, trans, emis, device=device)
    _, forward_backward_idx, forward_backward_accuracy = forward_backward_log_batch(test_x, test_y, trans, emis, device=device)
    print(f'word count weight, decode acc: [{decode_accuracy}], forward-backward acc: [{forward_backward_accuracy}]')

    trans = Variable(0.01 * torch.rand((T, T), dtype=torch.float32).to(device), requires_grad=True)
    emis = Variable(0.01 * torch.rand((T, W), dtype=torch.float32).to(device), requires_grad=True)

    # train
    print(f'training config: lr={LR}, max_iter={MAX_ITER}')
    optimizer = torch.optim.Adam([trans, emis], lr=LR)
    iter = 0
    start_time = time.time()
    for iter, (x,y) in enumerate(sampler(train_x, train_y, batchsize=BATCH_SIZE)):
        if iter>MAX_ITER:
            break
    
        ## testing
        if iter%PRINT_FREQ==0:
            with torch.no_grad():
                decode_idx, decode_accuracy = decode_batch(test_x, test_y, trans, emis, device=device)
                _, forward_backward_idx, forward_backward_accuracy = forward_backward_log_batch(test_x, test_y, trans, emis, device=device)
                print(f'iteration [{iter}/{MAX_ITER}], elapsed time [{time.strftime("%H:%M:%S", time.gmtime(time.time()-start_time))}], decode acc: [{decode_accuracy}], forward-backward acc: [{forward_backward_accuracy}]')
                decode_idx, decode_accuracy = decode_batch(train_x, train_y, trans, emis, device=device)
                _, forward_backward_idx, forward_backward_accuracy = forward_backward_log_batch(test_x, test_y, trans, emis, device=device)
                print(f'    training decode acc: [{decode_accuracy}], training forward-backward acc: [{forward_backward_accuracy}]')

        optimizer.zero_grad()
        logp_marginal, _, _ = forward_backward_log_batch(x, y, trans, emis, device=device)
        # p_marginal = logp_marginal.exp()
        p_marginal = logp_marginal.softmax(-1)
        p_marginal_y = torch.gather(p_marginal, 2, y.unsqueeze(-1))
        
        trans_logp = F.log_softmax(trans, 1)
        emis_logp = F.log_softmax(emis, 1)
        l1_reg = F.l1_loss(trans_logp, torch.zeros_like(trans_logp)) + F.l1_loss(emis_logp, torch.zeros_like(emis_logp))

        loss = - p_marginal_y.mean() + L1_REG * l1_reg # Minimum Bayes risk rule + l1 reg on weight

        loss.backward()
        optimizer.step()

