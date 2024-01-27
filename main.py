import torch
from utils.dataset import MuscimaDataset

if __name__ == "__main__":
    train_dataset = MuscimaDataset.load('data/default/train.pth')
    assert train_dataset[3] == ((1177, 263, 1297, 324), '71', (1200, 207, 1319, 3337), '34', True)