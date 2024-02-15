import os
import torch
from tqdm import tqdm
import random
import itertools
import numpy as np
from argparse import ArgumentParser
from xml.etree import ElementTree
from mung.io import read_nodes_from_file
import shutil
from omrdatasettools import Downloader, OmrDataset


def set_seed(seed):
    random.seed(seed)

from PIL import Image
import shutil
import os
from tqdm import tqdm

def generate(docs, ratio, clsname2id, dataset, mode):
    images_dir = f"yolo_object_detection/{dataset}/{mode}/images"
    labels_dir = f"yolo_object_detection/{dataset}/{mode}/labels"
    # Create (or clear) directories
    for directory in [images_dir, labels_dir]:
        os.makedirs(directory, exist_ok=True)  # Creates the directory if it doesn't exist, does nothing otherwise

    for doc in tqdm(docs, desc=f'Generating {mode}'):
        doc_name = doc[0].document

        doc_name = doc_name.replace("ideal", "symbol")

        
        src_path = f"yolo_object_detection/{dataset}/images/{doc_name}.png"
        dst_path = os.path.join(images_dir, f"{doc_name}.png")
        
        print(doc_name, src_path, dst_path )
        # Copy the image to the target file
        shutil.copy(src_path, dst_path)
        
        # Open the image to get its dimensions
        with Image.open(dst_path) as img:
            img_width, img_height = img.size
        
        # Open label file for writing
        with open(os.path.join(labels_dir, f"{doc_name}.txt"), "w") as f:
            for _, node in enumerate(doc):
                # exclude staff class 
                
                if node.class_name not in ["staffGrouping","staffSpace", "staffLine", "staff"]:
                    # Calculate normalized x_center, y_center, width, height
                    x_center = ((node.right - node.left) / 2 + node.left) / img_width
                    y_center = ((node.bottom - node.top) / 2 + node.top) / img_height
                    width = (node.right - node.left) / img_width
                    height = (node.bottom - node.top) / img_height
                    
                    # Write to label file
                    f.write(f"{clsname2id[node.class_name]} {x_center} {y_center} {width} {height}\n")
                else: 
                    print("Skip class: ", node.class_name)


import glob
import shutil
import os

def copy_and_rename_images(source_dir, target_dir):
    """
    Copies all .png files from source directories matching the pattern to the target directory
    with a new naming convention.

    Parameters:
    - source_dir: The source directory containing the image files.
    - target_dir: The target directory where the renamed image files will be copied.
    """
    # Create the target directory if it does not exist
    os.makedirs(target_dir, exist_ok=True)
    
    # Pattern to match all PNG files in the specified source directory structure
    pattern = os.path.join(source_dir, 'ideal', 'w-*', 'symbol', '*.png')
    
    for file_path in glob.glob(pattern):
        # Extract parts of the file path
        parts = file_path.split(os.sep)
        
        # Extract the writer number 'w-xx'
        writer = parts[-3]
        
        # Extract the page number 'pXXX.png'
        page = parts[-1].split('.')[0][-2:]
        
        # Construct the symbol file name
        new_file_name = f'CVC-MUSCIMA_W-{writer.split("-")[1]}_N-{page}_D-symbol.png'

        # Construct the ideal file name
        ideal_file_name = f'CVC-MUSCIMA_W-{writer.split("-")[1]}_N-{page}_D-ideal.png'

        annotated_image_path = f"yolo_object_detection/{dataset}/images/"


        print(os.path.join(annotated_image_path, ideal_file_name))
        if os.path.exists(os.path.join(annotated_image_path, ideal_file_name)):

        
            # Construct the full target file path
            target_file_path = os.path.join(target_dir, new_file_name)
            
            # Copy and rename the file
            shutil.copy(file_path, target_file_path)
            print(f'Copied and renamed {file_path} to {target_file_path}')



if __name__ == "__main__":
    # # Build images folder
    # source_dir = 'data/MUSCIMA++/CvcMuscima-Distortions'
    # target_dir = 'yolo_object_detection/datasets_staff_removed/images'  # Current directory
    # copy_and_rename_images(source_dir, target_dir)

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


    set_seed(args.seed)

    # downloader = Downloader()
    # downloader.download_and_extract_dataset(MuscimaPlusPlus_Images, "data")

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

    print("Train...")
    generate(train_docs, args.ratio, clsname2id, "datasets_staff_removed", "train")
    print("Val...")
    generate(val_docs, 'val', clsname2id, "datasets_staff_removed", "valid")
    print("Test...")
    generate(test_docs, 'test', clsname2id, "datasets_staff_removed",  "test")

    
    print("DONE.")

  