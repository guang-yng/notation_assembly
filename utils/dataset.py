import os
import itertools
import torch
from torch.utils.data import Dataset
from muscima.io import parse_cropobject_list

class MuscimaDataset(Dataset):
    r"""
    MuscimaDataset class is a torch.utils.data.Dataset subclass, 
    iterating all pairs of symbols and outputs the bounding boxes, classnames, and whether an edge exists.
    """
    def __init__(self, dataset, split='train'):
        self.dataset = dataset
        self.split = split

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

    def save(self, path):
        torch.save((self.dataset, self.split), path)

    @classmethod
    def load(cls, path):
        return cls(*torch.load(path))