import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import numpy as np
from argparse import ArgumentParser
from mung.io import read_nodes_from_file
from dataset import MuscimaDataset

def set_seed(seed):
    random.seed(seed)

def draw_and_save(docs, dir, margin=1, file_extension='pdf'):
    for i, doc in enumerate(tqdm(docs, desc='Drawing')):
        top = min([c.top for c in doc])
        left = min([c.left for c in doc])
        bottom = max([c.bottom for c in doc])
        right = max([c.right for c in doc])
        height = bottom - top + 2 * margin
        width = right - left + 2 * margin
        canva = np.zeros((height, width), dtype='uint8')
        for c in doc:
            _pt = c.top - top + margin
            _pl = c.left - left + margin
            canva[_pt:_pt+c.height, _pl:_pl+c.width] += c.mask if c.class_name != 'staffSpace' else 1 - c.mask
            
        canva[canva > 0] = 1
        plt.imshow(canva, cmap='gray', interpolation='nearest')
        plt.savefig(os.path.join(dir, f"{i}.{file_extension}"))

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-d', '--data', 
                        default='data/MUSCIMA++/v2.0/data/annotations',
                        help="data directory")
    parser.add_argument('--save_dir', default='data/default', help='The output directory')
    parser.add_argument('--seed', default=314, help='random seed')
    parser.add_argument('--file_extension', type=str, choices=['pdf', 'png'])
                        
    args = parser.parse_args()

    for split in ['train', 'val', 'test']:
        if os.path.exists(os.path.join(args.save_dir, f'{split}_images')):
            raise ValueError(f"The save_dir/{split}_images exists!")
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    for split in ['train', 'val', 'test']:
        os.mkdir(os.path.join(args.save_dir, f'{split}_images'))

    set_seed(args.seed)

    cropobject_fnames = [os.path.join(args.data, f) for f in os.listdir(args.data) if f.endswith('xml')]
    docs = [read_nodes_from_file(f) for f in cropobject_fnames]
    random.shuffle(docs)
    train_docs = docs[:int(0.6*len(docs))]
    val_docs = docs[int(0.6*len(docs)): int(0.8*len(docs))]
    test_docs = docs[int(0.8*len(docs)):]

    draw_and_save(train_docs, os.path.join(args.save_dir, 'train_images'), file_extension=args.file_extension)
    draw_and_save(val_docs, os.path.join(args.save_dir, 'val_images'), file_extension=args.file_extension)
    draw_and_save(test_docs, os.path.join(args.save_dir, 'test_images'), file_extension=args.file_extension)