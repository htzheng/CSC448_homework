import sys
import numpy as np
NINF = -1e9

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
    W = len(words) + 1   # add an out-of-vocabulary word in the end
    Trans = np.zeros((T, T), dtype=np.float32)
    Emis = np.zeros((T, W), dtype=np.float32)
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
        seq, label = data_i, None
        data.append(seq)
        data_idx.append([words_idx.get(w, W-1) for w in seq])  # if word not found in word list, return the last index
        data_label.append(label)

    def get_tag(pre, i, t):
        return get_tag(pre, i-1, pre[i,t]) + [t] if i>0 else []
        
    def veterbi_decode(seq, label):
        N = len(seq)
        delta = NINF * np.ones((N + 1, T), dtype=np.float32)
        pre = np.zeros((N + 1, T), dtype=np.int32)

        delta[0, T-1] = 0
        for i in range(1, N+1):
            x = words_idx.get(seq[i-1], W-1)
            for t in range(T):
                values = delta[i-1,:] + Trans[:,t] + Emis[t, x]
                delta[i,t] = np.max(values)
                pre[i,t] = np.argmax(values)

        max_idx = np.argmax(delta[N, :])
        max_value = delta[N, max_idx]
        tag_idx = get_tag(pre, N, max_idx)
        tag = [pos[idx] for idx in tag_idx]
        
        return " ".join([str(max_value.item())] + tag)

    for seq, label in zip(data, data_label):
        print(veterbi_decode(seq, label))
        
