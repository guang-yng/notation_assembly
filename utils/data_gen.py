import os
import torch
from tqdm import tqdm
import random
import itertools
import numpy as np
from argparse import ArgumentParser
from xml.etree import ElementTree
from mung.io import read_nodes_from_file
from dataset import MuscimaDataset

def set_seed(seed):
    random.seed(seed)

def generate_train(docs, ratio, clsname2id):
    dataset = []
    for doc in tqdm(docs, desc='Generating train'):
        pos_pairs = set()
        n_sym = len(doc)
        id2idx = {c.id: i for i, c in enumerate(doc)}
        for i, c in enumerate(doc):
            for jid in c.outlinks:
                j = id2idx[jid]
                pos_pairs.add((i, j) if i < j else (j, i))
        neg_pairs = set((i, j) for j in range(1, n_sym) for i in range(j)) - pos_pairs

        if ratio is not None:
            assert ratio > 0 and ratio <= 100
            num_neg_pairs = int((100-ratio)/ratio * len(pos_pairs))
            if num_neg_pairs > len(neg_pairs):
                print("Warning! The ratio is smaller than natural percentage.")
                num_neg_pairs = len(neg_pairs)
            neg_pairs = set(random.sample(list(neg_pairs), num_neg_pairs))
        for i, j in pos_pairs:
            dataset.append((doc[i].bounding_box, clsname2id[doc[i].class_name], doc[j].bounding_box, clsname2id[doc[j].class_name], True))
        for i, j in neg_pairs:
            dataset.append((doc[i].bounding_box, clsname2id[doc[i].class_name], doc[j].bounding_box, clsname2id[doc[j].class_name], False))
            
    return MuscimaDataset(dataset)

def generate_test(docs, split, clsname2id):
    dataset = []
    for doc in tqdm(docs, desc=f'Generating {split}'):
        n_sym = len(doc)
        graph = np.zeros((n_sym, n_sym), dtype=np.int32)
        symbols = []
        id2idx = {c.id: i for i, c in enumerate(doc)}
        for i, c in enumerate(doc):
            for jid in c.outlinks:
                j = id2idx[jid]
                graph[i][j] = 1
                graph[j][i] = 1
            symbols.append((doc[i].bounding_box, clsname2id[doc[i].class_name]))
        dataset.append((symbols, graph))
    return MuscimaDataset(dataset, split=split)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-d', '--data', 
                        default='data/MUSCIMA++/v2.0/data/annotations',
                        help="data directory")
    parser.add_argument('-r', '--ratio',
                        default=None, help='the percentage of positive pairs in training data, default to natural percentage')
    parser.add_argument('--classes', default='data/MUSCIMA++/v2.0/specifications/mff-muscima-mlclasses-annot.xml',
                        help='the path to the musima classes definition xml file')
    parser.add_argument('--save_dir', default='data/default', help='The output directory')
    parser.add_argument('--seed', default=314, help='random seed')
                        
    args = parser.parse_args()

    if os.path.exists(args.save_dir):
        raise ValueError("The save_dir exists!")
    os.mkdir(args.save_dir)

    set_seed(args.seed)

    cropobject_fnames = [os.path.join(args.data, f) for f in os.listdir(args.data) if f.endswith('xml')]
    docs = [read_nodes_from_file(f) for f in cropobject_fnames]
    random.shuffle(docs)
    train_docs = docs[:int(0.6*len(docs))]
    val_docs = docs[int(0.6*len(docs)): int(0.8*len(docs))]
    test_docs = docs[int(0.8*len(docs)):]

    clsname2id = {}
    tree = ElementTree.parse(args.classes)
    root = tree.getroot()
    for nodeclass in root:
        clsname = nodeclass.find('Name').text
        idx = nodeclass.find('Id').text
        clsname2id[clsname] = idx


    train_dataset = generate_train(train_docs, args.ratio, clsname2id)
    val_dataset = generate_test(val_docs, 'val', clsname2id)
    test_dataset = generate_test(test_docs, 'test', clsname2id)

    print("Saving...")
    train_dataset.save(os.path.join(args.save_dir, 'train.pth'))
    val_dataset.save(os.path.join(args.save_dir, 'val.pth'))
    test_dataset.save(os.path.join(args.save_dir, 'test.pth'))
    print("DONE.")