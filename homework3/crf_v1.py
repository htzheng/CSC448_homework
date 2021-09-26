import sys
import os
import random
from collections import OrderedDict
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm


NINF = -1e9
MAX_ITER = 2000
MAX_LENGTH = 10
START = '->'
START_TAG = '->'
START_IDX = 1
END='.'
END_TAG='.'
END_IDX = 0

LR = 0.002

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

def to_index(data, index_table):
    N, L = len(data), len(data[0])
    m = torch.zeros((N, L), dtype=torch.int64)
    for i in range(N):
        for j in range(L):
            m[i,j] = index_table[data[i][j]]
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
        # print(t)
        # print(pre[:,i])
        # print(torch.gather(pre[:,i], 1, t))
        # print(torch.index_select(pre[:,i], 1, t))
        # pre[:,i][0,t[0]], pre[:,i][1,t[1]]
        

        return torch.cat((get_tag_batched(pre, i-1, torch.gather(pre[:,i], 1, t.unsqueeze(1)).squeeze(1)), t.unsqueeze(1)), dim=1)
    else:
        return torch.full_like(t, START_IDX, dtype=torch.int64).unsqueeze(1)

''' decoding '''
def get_tag(pre, i, t):
    return get_tag(pre, i-1, pre[i,t]) + [t.item()] if i>0 else [START_IDX]

def decode(x, y, trans, emis):
    N, = x.size()
    T, W = emis.size()

    # B, N = x.size()
    # T, W = emis.size()
    # print(f'training data: words {x}\ntraining data: tags {y}')
    
    # 
    # x = x[0,...]
    # y = y[0,...]

    ## softmax
    # trans_p = torch.softmax(trans, 1)
    # emis_p = torch.softmax(emis, 1)

    # trans = trans[None,:,:].expand(B,-1,-1)
    # emis = emis[None,:,:].expand(B,-1,-1)

    delta = NINF * torch.ones((N, T), dtype=torch.float32)
    pre = torch.zeros((N, T), dtype=torch.int64)
    delta[0,START_IDX] = 0
    for i in range(1, N):
        x_i = x[i]
        # x_i = x[:, i:i+1] # B x 1
        # x_i_onehot = torch.zeros((B, W), dtype=torch.float32)
        # x_i_onehot.scatter_(1, x_i, 1)

        for t in range(T):
            # values = delta[:, i-1,:] + trans[None,:,t] + (emis[None,...] @ x_i_onehot[...,None])[:,:,0]
            # delta[:,i,t], pre[:,i,t] = torch.max(values,1)
            values = delta[i-1,:] + trans[:,t] + emis[t,x_i]
            delta[i,t], pre[i,t] = torch.max(values), torch.argmax(values)

        # x_i_onehot = torch.zeros((B, W), dtype=torch.float32).scatter_(1, x_i, 1)
        # b1 = (emis[None,...] @ x_i_onehot[...,None])[:,:,0]

    max_idx = torch.argmax(delta[N-1, :])
    max_value = delta[N-1, max_idx]
    tag_idx = get_tag(pre, N-1, max_idx)
    tag = [tags[idx] for idx in tag_idx]

    print(max_value, tag_idx)
    print(y)
    return

def forward_backward(x, y, trans, emis, trans_p, emis_p):
    N, = x.size()
    T, W = emis_p.size()

    ''' forward '''
    a = torch.zeros((N, T), dtype=torch.float32)
    a[0,START_IDX] = 1
    for i in range(1, N):
        x_i = x[i]
        for t in range(T):
            # slow version
            for tp in range(T):
                a[i,t] = a[i,t] + emis_p[t,x_i] * trans_p[tp,t] * a[i-1,tp]
                # a[:,i,t] = a[:,i,t] + emis[None,t,x_i] * trans[None,tp,t] * a[:,i-1,tp]
                # a[:,i,t] = a[:,i,t] + 1 * trans[None,tp,t] * a[:,i-1,tp]

            # value1 = a[:,i,t].clone()
            # # value2 = emis[t,x_i][:,None] * torch.matmul(trans, a[:,i-1,:,None]).squeeze(-1)
            # value2 = 1 * torch.matmul(trans, a[:,i-1,:,None]).squeeze(-1)
            # value2 = value2.sum(1)
            # print(value1[0,...])
            # print(value2[0,...])
            # print('---')
        # print(a[i,:])
        # print('a---')

    ''' backward '''
    b = torch.zeros((N, T), dtype=torch.float32)
    b[N-1,:] = 1/a[N-1,:].sum()
    for i in range(N-2,-1,-1):
        x_i1 = x[i+1]
        for t in range(T):
            # slow version
            for tp in range(T):
                b[i,t] = b[i,t] + emis_p[tp,x_i1] * trans_p[t,tp] * b[i+1,tp]
        # print(b[i,:])
        # print('b---')

    ## TODO: compute marginal probablity, verify if max value is the label GT
    ab = a*b
    _, tag_idx = torch.max(ab, 1)
    print(tag_idx)
    print(y)
    return a, b

def decode_batch(x, y, trans_logp, emis_logp, device='cpu'):
    B, N = x.size()
    T, W = emis.size()

    ''' decode batched '''
    delta = NINF * torch.ones((B, N, T), dtype=torch.float32).to(device)
    pre = torch.zeros((B, N, T), dtype=torch.int64).to(device)
    delta[:,0,START_IDX].fill_(0)
    for i in range(1, N):
        x_i = x[:,i]
        ''' fast version'''
        for t in range(T):
            values = delta[:,i-1,:] + trans[:,t].unsqueeze(0) + emis[t,x_i].unsqueeze(1)
            delta[:,i,t], pre[:,i,t] = torch.max(values,1)
        ''' faster version'''
        values = delta[:,i-1,:].unsqueeze(-1) + trans[:,:].unsqueeze(0) + emis[:,x_i].transpose(0,1).unsqueeze(1)
        delta[:,i,:], pre[:,i,:] = torch.max(values,1)
    _, max_idx = torch.max(delta[:,N-1, :], 1)
    decode_idx = get_tag_batched(pre, N-1, max_idx)

    print(f'----------joint max acc: {(decode_idx==y).float().mean()}---------------')
    for bs in range(B):
        print(decode_idx[bs,...])
        print(y[bs,...])
    print()
    return decode_idx

def forward_backward_log_batch(x, y, trans_logp, emis_logp, device='cpu'):
    B, N = x.size()
    T, W = emis.size()

    # trans_p = torch.softmax(trans, 1)
    # emis_p = torch.softmax(emis, 1)
    # trans[:,START_IDX] = trans[:,START_IDX] + NINF
    # emis[START_IDX,:] = emis[START_IDX,:] + NINF
    # trans_logp = F.log_softmax(trans, 1)
    # emis_logp = F.log_softmax(emis, 1)
    # trans_logp = trans
    # emis_logp = emis

    ''' decode batched '''
    # delta = NINF * torch.ones((B, N, T), dtype=torch.float32)
    # pre = torch.zeros((B, N, T), dtype=torch.int64)
    # delta[:,0,START_IDX].fill_(0)
    # for i in range(1, N):
    #     x_i = x[:,i]
    #     ''' fast version'''
    #     for t in range(T):
    #         values = delta[:,i-1,:] + trans[:,t].unsqueeze(0) + emis[t,x_i].unsqueeze(1)
    #         delta[:,i,t], pre[:,i,t] = torch.max(values,1)
    #     ''' faster version'''
    #     values = delta[:,i-1,:].unsqueeze(-1) + trans[:,:].unsqueeze(0) + emis[:,x_i].transpose(0,1).unsqueeze(1)
    #     delta[:,i,:], pre[:,i,:] = torch.max(values,1)
    # _, max_idx = torch.max(delta[:,N-1, :], 1)
    # decode_idx = get_tag_batched(pre, N-1, max_idx)

    # print('----------joint max---------------')
    # for bs in range(B):
    #     print(decode_idx[bs,...])
    #     print(y[bs,...])

    ''' forward batched '''
    a = torch.zeros((B, N, T), dtype=torch.float32).to(device)
    a[:,0,START_IDX] = 1
    log_a = torch.zeros((B, N, T), dtype=torch.float32).to(device)
    log_a[:,0,:] = NINF
    log_a[:,0,START_IDX] = 0

    for i in range(1, N):
        x_i = x[:,i]
        for t in range(T):
            ''' slow version '''
            for tp in range(T):
                log_a[:,i,t] = log_a[:,i,t] + (emis_logp[t,x_i] + log_a[:,i-1,tp] + trans_logp[tp,t]).exp()
            log_a[:,i,t] = log_a[:,i,t].log()
            ''' fast version '''
            log_a[:,i,t] = (emis_logp[t,x_i].unsqueeze(1) + log_a[:,i-1,:] + trans_logp[:,t].unsqueeze(0)).logsumexp(1) #10           10x1      10x46     1x46   (10x46)
        ''' faster version '''
        z3 = (emis_logp[:,x_i].transpose(0,1).unsqueeze(1) + log_a[:,i-1,:].unsqueeze(2) + trans_logp.unsqueeze(0)).logsumexp(1) # 10x46                10x1x46                 10x46x1            1x46x46
    a = log_a.exp()

    ''' backward batched '''
    b = torch.zeros((B, N, T), dtype=torch.float32).to(device)
    b[:,N-1,:] = 1
    log_b = torch.zeros((B, N, T), dtype=torch.float32).to(device)
    log_b[:,N-1,:] = 0
    for i in range(N-2,-1,-1):
        x_i1 = x[:,i+1]
        for t in range(T):
            ''' slow version '''
            # for tp in range(T):
            #     # b[:,i,t] = b[:,i,t] + emis_p[tp,x_i1] * trans_p[t,tp] * b[:,i+1,tp]
            #     log_b[:,i,t] = log_b[:,i,t] + (emis_logp[tp,x_i1] + trans_logp[t,tp] + log_b[:,i+1,tp]).exp()
            # log_b[:,i,t] = log_b[:,i,t].log()
            # z1 = log_b[:,i,t]
            # print(z1)
            ''' fast version '''
            # log_b[:,i,t] = (emis_logp[:,x_i1].transpose(0,1) + trans_logp[t,:].unsqueeze(0) + log_b[:,i+1,:]).logsumexp(1)    # 10 =  10x46   1x46    10x46 (10x46)
            
        ''' faster version ''' #b[:,i,:]
        b[:,i,:] = (emis_logp[:,x_i1].transpose(0,1).unsqueeze(-1) + trans_logp.transpose(0,1).unsqueeze(0) + log_b[:,i+1,:].unsqueeze(-1)).logsumexp(1)
            # 10x46 = 10x46x1   1x46x46    10x1 (10x46x46)  

    log_marginal = log_a+log_b
    # print('----------marginal max (log)------------------')
    # for bs in range(B):
    #     _, tag_idx = torch.max(log_marginal[bs,...], 1)
    #     print(tag_idx)
    #     print(y[bs,...])

    return log_marginal


def forward_backward_batch(x, y, trans, emis, test_a=None, test_b=None):
    B, N = x.size()
    T, W = emis.size()

    trans_p = torch.softmax(trans, 1)
    emis_p = torch.softmax(emis, 1)
    trans[:,START_IDX] = trans[:,START_IDX] + NINF
    emis[START_IDX,:] = emis[START_IDX,:] + NINF
    ''' decode batched '''
    # delta = NINF * torch.ones((B, N, T), dtype=torch.float32)
    # pre = torch.zeros((B, N, T), dtype=torch.int64)
    # delta[:,0,START_IDX].fill_(0)
    # for i in range(1, N):
    #     x_i = x[:,i]
    #     ''' fast version'''
    #     for t in range(T):
    #         values = delta[:,i-1,:] + trans[:,t].unsqueeze(0) + emis[t,x_i].unsqueeze(1)
    #         delta[:,i,t], pre[:,i,t] = torch.max(values,1)
    #     ''' faster version'''
    #     values = delta[:,i-1,:].unsqueeze(-1) + trans[:,:].unsqueeze(0) + emis[:,x_i].transpose(0,1).unsqueeze(1)
    #     delta[:,i,:], pre[:,i,:] = torch.max(values,1)
    # _, max_idx = torch.max(delta[:,N-1, :], 1)
    # decode_idx = get_tag_batched(pre, N-1, max_idx)

    # print('----------joint max---------------')
    # for bs in range(B):
    #     print(decode_idx[bs,...])
    #     print(y[bs,...])

    ''' forward batched '''
    a = torch.zeros((B, N, T), dtype=torch.float32)
    a[:,0,START_IDX] = 1
    for i in range(1, N):
        x_i = x[:,i]
        for t in range(T):
            ''' slow version '''
            # for tp in range(T):
            #     a[:,i,t] = a[:,i,t] + emis_p[t,x_i] * a[:,i-1,tp] * trans_p[tp,t] #.unsqueeze(0)
            #     # for bs in range(B):
            #     #     x_i = x[bs,i]
            #     #     a[bs,i,t] = a[bs,i,t] + emis_p[t,x_i] * trans_p[tp,t] * a[bs,i-1,tp]
            ''' fast version '''
            a[:,i,t] = emis_p[t,x_i] * (a[:,i-1,:] @ trans_p[:,t])
        ''' faster version '''
        a[:,i,:] = emis_p[:,x_i].transpose(0,1) * (a[:,i-1,:].unsqueeze(1) @ trans_p[:,:]).squeeze(1)

    ''' backward batched '''
    b = torch.zeros((B, N, T), dtype=torch.float32)
    b[:,N-1,:] = 1
    for i in range(N-2,-1,-1):
        x_i1 = x[:,i+1]
        for t in range(T):
            ''' slow version '''
            # for tp in range(T):
            #     b[:,i,t] = b[:,i,t] + emis_p[tp,x_i1] * trans_p[t,tp] * b[:,i+1,tp]
                # for bs in range(B):
                #     x_i1 = x[bs,i+1]
                #     b[bs,i,t] = b[bs,i,t] + emis_p[tp,x_i1] * trans_p[t,tp] * b[bs,i+1,tp]
            # print(b[0,i,t]- test_b[i,t])
            ''' fast version '''
            # b[:,i,t] = (emis_p[:,x_i1].transpose(0,1) * b[:,i+1,:]) @ trans_p[t,:]
        ''' faster version '''
        b[:,i,:] = (emis_p[:,x_i1].transpose(0,1) * b[:,i+1,:]) @ trans_p.transpose(0,1)

    ## TODO: compute marginal probablity, verify if max value is the label GT
    marginal = a*b
    print('----------marginal max------------------')
    for bs in range(B):
        _, tag_idx = torch.max(marginal[bs,...], 1)
        print(tag_idx)
        print(y[bs,...])

    return a


def word_count(train_x, train_y, cache='weight_count.pkl'):
    if os.path.isfile(cache):
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
    words = OrderedDict([(END, 0), (START, 0)])
    tags = OrderedDict([(END_TAG, 0), (START_TAG, 0)])
    for x, y in zip(train_x_, train_y_):
        for x_i, y_i in zip(x, y):
            words.update({x_i:0})
            tags.update({y_i:0})
    for x, y in zip(test_x_, test_y_):
        for x_i, y_i in zip(x, y):
            words.update({x_i:0})
            tags.update({y_i:0})
    tags, words = list(tags), list(words)
    T, W = len(tags), len(words)
    tags_idx = {name:idx for (idx,name) in enumerate(tags)}
    words_idx = {name:idx for (idx,name) in enumerate(words)}
    
    ## convert tags/words to index
    train_x, train_y = to_index(train_x_, words_idx).to(device), to_index(train_y_, tags_idx).to(device)
    test_x, test_y = to_index(test_x_, words_idx).to(device), to_index(test_y_, tags_idx).to(device)

    ## weight by word count
    trans, emis = word_count(train_x, train_y)
    trans, emis = trans.to(device), emis.to(device)

    ## test forward-backward
    # bs = 10
    # for i in range(bs):
    #     # decode(train_x[i], train_y[i], trans, emis)
    #     forward_backward(train_x[i], train_y[i], trans, emis, trans_p, emis_p)
    # test forward-backward batch
    # p_marginal = forward_backward_batch(train_x[0:bs,...], train_y[0:bs,...], trans, emis)
    # p_marginal = forward_backward_log_batch(train_x[0:bs,...], train_y[0:bs,...], trans, emis)

    # init trans and emis matrix
    # trans, emis = word_count(train_x, train_y)
    # trans = Variable(trans, requires_grad=True)
    # emis = Variable(emis, requires_grad=True)
    trans = Variable(0.01 * torch.rand((T, T), dtype=torch.float32).to(device), requires_grad=True)
    emis = Variable(0.01 * torch.rand((T, W), dtype=torch.float32).to(device), requires_grad=True)

    # train
    optimizer = torch.optim.Adam([trans, emis], lr=0.001)
    iter = 0
    for iter, (x,y) in enumerate(sampler(train_x[0:10,...], train_y[0:10,...], batchsize=10)):
        if iter>MAX_ITER:
            break
    
        ## testing
        if iter%10==0:
            with torch.no_grad():
                trans_logp = F.log_softmax(trans, 1)
                emis_logp = F.log_softmax(emis, 1)
                decode_idx = decode_batch(train_x[0:10], train_y[0:10], trans_logp, emis_logp, device=device)
                print()
            
        optimizer.zero_grad()
        trans_logp = F.log_softmax(trans, 1)
        emis_logp = F.log_softmax(emis, 1)
        logp_marginal = forward_backward_log_batch(x, y, trans_logp, emis_logp, device=device)
        p_marginal = logp_marginal.softmax(-1)
        p_marginal_y = torch.gather(p_marginal, 2, y.unsqueeze(-1))
        loss = - p_marginal_y.mean() # Minimum Bayes risk rule

        print(loss)
        loss.backward()
        optimizer.step()

