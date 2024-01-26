import torch
from utils.dataset import MuscimaDataset

if __name__ == "__main__":
    train_dataset = MuscimaDataset.load('data/default/train.pth')
    assert train_dataset[3] == ((1746, 708, 1769, 725), '0', (790, 208, 823, 3337), '32', True)