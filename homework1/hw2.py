import sys
import torch
NINF = -1e9
HAS_LABEL = False
OO, XX = 0., 0.

if __name__ == '__main__':
    weights = dict()
    with open(sys.argv[1], 'r') as f:
        for line in f:
            key, value = line.strip().split(' ')
            weights[key] = float(value)

    data_ = []
    with open(sys.argv[2], 'r') as f:
        for line in f:
            line = line.strip().split(' ')
            data_.append(line[1:])

    # get all tags and words
    pos = set()
    words = set()
    for item in weights.keys():
        if item.startswith('T_'):
            _, a, b = item.split('_')
            pos.add(a)
            pos.add(b)
        elif item.startswith('E_'):    
            _, a, w = item.split('_')
            pos.add(a)
            words.add(w)
    pos, words = list(pos), list(words)
    pos_idx = {name:idx for (idx,name) in enumerate(pos)}
    words_idx = {name:idx for (idx,name) in enumerate(words)}

    # init trans and emis matrix
    T = len(pos) + 1     # add START symbol in the end
    W = len(words) + 1   # add an out-of-vocabulary symbol in the end
    Trans = torch.zeros((T, T), dtype=torch.float32)
    Emis = torch.zeros((T, W), dtype=torch.float32)
    Trans[:,-1] = NINF
    Emis[-1,:] = NINF  # init trans and emis for START symbol

    for item in weights.keys():
        if item.startswith('T_'):
            _, a, b = item.split('_')
            Trans[pos_idx[a], pos_idx[b]] = weights[item]
        elif item.startswith('E_'):    
            _, a, w = item.split('_')
            Emis[pos_idx[a], words_idx[w]] = weights[item]

    # convert data to index
    data = []
    data_idx = []
    data_label = []
    for data_i in data_:
        if not HAS_LABEL:
            seq, label = data_i, None
        else:
            seq, label = data_i[0::2], data_i[1::2]
        data.append(seq)
        data_idx.append([words_idx.get(w, W-1) for w in seq])  # if word not found in word list, return the last index
        data_label.append(label)

    def veterbi_decode(seq, label):
        global OO, XX

        N = len(seq)
        delta = NINF * torch.ones((N + 1, T), dtype=torch.float32)
        pre = torch.zeros((N + 1, T), dtype=torch.int32)

        def get_tag(pre, i, t):
            return get_tag(pre, i-1, pre[i,t]) + [t] if i>0 else []

        delta[0, T-1] = 0
        for i in range(1, N+1):
            x = words_idx.get(seq[i-1], W-1)
            for t in range(T):
                ## slow version
                # for tp in range(T):
                #     value = delta[i-1, tp] + Trans[tp,t] + Emis[t, x]
                #     if value >= delta[i, t]:
                #         delta[i, t] = value
                #         pre[i,t] = tp

                ## fast version
                values = delta[i-1,:] + Trans[:,t] + Emis[t, x]
                delta[i,t] = torch.max(values)
                pre[i,t] = torch.argmax(values)

        max_idx = torch.argmax(delta[N, :])
        max_value = delta[N, max_idx]
        tag_idx = get_tag(pre, N, max_idx)
        tag = [pos[idx] for idx in tag_idx]
        
        if label is not None:
            oo = sum([x==y for (x,y) in zip(tag, label)])
            xx = sum([x!=y for (x,y) in zip(tag, label)])
            OO+=oo
            XX+=xx
            print(label)
            print(OO/(OO+XX))
        else:
            oo, xx = 0, 0
        return " ".join([str(max_value.item())] + tag), oo, xx

    for seq, label in zip(data, data_label):
        s, oo, xx = veterbi_decode(seq, label)
        print(s)
        
