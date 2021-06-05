import os
import time

import ujson as json
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from itertools import tee 

class MySet(Dataset):
    def __init__(self):
        super(MySet, self).__init__()
        self.content = open('./json/json').readlines()

        indices = np.arange(len(self.content))
        val_indices = np.random.choice(indices, len(self.content) // 5)

        self.val_indices = set(val_indices.tolist())

    def __len__(self):
        return len(self.content)

    def __getitem__(self, idx):
        rec = json.loads(self.content[idx])
        if idx in self.val_indices:
            rec['is_train'] = 0
        else:
            rec['is_train'] = 1
        return rec
    
    def getitem(self, idx):
        rec = json.loads(self.content[idx])
        if idx in self.val_indices:
            rec['is_train'] = 0
        else:
            rec['is_train'] = 1
        return rec

def collate_fn(recs):
    
    itr=tee(recs,4)
    
    forward = map(lambda x: x['forward'], itr[0])
    backward = map(lambda x: x['backward'], itr[1])

    def to_tensor_dict(recs):
        itr=tee(recs,6)
        values = torch.FloatTensor(list(map(lambda r: r['values'], itr[0])))
        masks = torch.FloatTensor(list(map(lambda r: r['masks'], itr[1])))
        deltas = torch.FloatTensor(list(map(lambda r: r['deltas'], itr[2])))

        evals = torch.FloatTensor(list(map(lambda r: r['evals'], itr[3])))
        eval_masks = torch.FloatTensor(list(map(lambda r: r['eval_masks'], itr[4])))
        forwards = torch.FloatTensor(list(map(lambda r: r['forwards'], itr[5])))


        return {'values': values, 'forwards': forwards, 'masks': masks, 'deltas': deltas, 'evals': evals, 'eval_masks': eval_masks}

    ret_dict = {'forward': to_tensor_dict(forward), 'backward': to_tensor_dict(backward)}

    ret_dict['labels'] = torch.IntTensor(list(map(lambda x: x['label'], itr[2])))
    ret_dict['is_train'] = torch.IntTensor(list(map(lambda x: x['is_train'], itr[3])))

    return ret_dict

def get_loader(batch_size = 64, shuffle = True):
    data_set = MySet()
    data_iter = DataLoader(dataset = data_set, \
                              batch_size = batch_size, \
                              num_workers = 4, \
                              shuffle = shuffle, \
                              pin_memory = True, \
                              collate_fn = collate_fn
    )

    return data_iter
