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

CLASSNAME2ID = {
    "noteheadHalf": 0, "noteheadWhole": 0, 
    "noteheadFull": 1,
    "stem": 2,
    "beam": 3,
    "legerLine": 4,
    "augmentationDot": 5,
    "slur": 6,
    "rest8th": 7,
    "accidentalNatural": 8,
    "accidentalFlat": 9,
    "accidentalSharp": 10,
    "barline": 11, 
    "gClef": 12,
    "fClef": 13,
    "dynamicLetterP": 14,
    "dynamicLetterM": 15,
    "dynamicLetterF": 16,
    "keySignature": 17,
    "flag8thUp": 18,
    "flag8thDown": 19,
}

def set_seed(seed):
    random.seed(seed)

from PIL import Image
import shutil
import os
from tqdm import tqdm

def generate(docs, clsname2id, save_dir, mode, crop_times = 0, image_source_dir = "MUSCIMA++/datasets_r_staff/images"):
    images_dir = os.path.join(save_dir, mode, 'images')
    labels_dir = os.path.join(save_dir, mode, 'labels')
    # Create (or clear) directories
    for directory in [images_dir, labels_dir]:
        os.makedirs(directory, exist_ok=True)  # Creates the directory if it doesn't exist, does nothing otherwise

    for doc in tqdm(docs, desc=f'Generating {mode}'):
        doc_name = doc[0].document

        doc_name = doc_name.replace("ideal", "symbol")

        
        src_path = os.path.join(image_source_dir, f"{doc_name}.png")
        
        if crop_times == 0:
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
                    
                    if node.class_name in clsname2id:
                        # Calculate normalized x_center, y_center, width, height
                        x_center = ((node.right - node.left) / 2 + node.left) / img_width
                        y_center = ((node.bottom - node.top) / 2 + node.top) / img_height
                        width = (node.right - node.left) / img_width
                        height = (node.bottom - node.top) / img_height
                        
                        # Write to label file
                        f.write(f"{clsname2id[node.class_name]} {x_center} {y_center} {width} {height}\n")
                    else: 
                        print("Skip class: ", node.class_name)
        else:
            with Image.open(src_path) as img:
                source_img_width, source_img_height = img.size
            crops = []
            if mode != 'test':
                for _ in range(crop_times):
                    x = random.randint(0, source_img_width-1216)
                    y = random.randint(0, source_img_height-1216)
                    crops.append((x, y))
            else:
                for x in range(0, source_img_width, 1216):
                    for y in range(0, source_img_height, 1216):
                        _x, _y = min(x, source_img_width-1216), min(y, source_img_height-1216)
                        crops.append((_x, _y))
                
            for times, crop in enumerate(crops):
                x, y = crop
                dst_path = os.path.join(images_dir, f"{doc_name}_{times}.png")
                with Image.open(src_path) as img:
                    img = img.crop((x, y, x+1216, y+1216))
                    img.resize((608, 608))
                    img.save(dst_path)
                with open(os.path.join(labels_dir, f"{doc_name}_{times}.txt"), "w") as f:
                    for _, node in enumerate(doc):
                        # exclude staff class 
                        if node.bottom > y+1216 or node.top < y or node.left < x or node.right > x+1216:
                            continue
                        if node.class_name in clsname2id:
                            # Calculate normalized x_center, y_center, width, height
                            x_center = (((node.right - node.left) / 2 + node.left)-x) / 1216
                            y_center = (((node.bottom - node.top) / 2 + node.top)-y) / 1216
                            width = (node.right - node.left) / 1216
                            height = (node.bottom - node.top) / 1216
                            
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
                        default='MUSCIMA++/v2.0/data/annotations',
                        help="data directory")
    parser.add_argument('--classes', default='MUSCIMA++/v2.0/specifications/mff-muscima-mlclasses-annot.xml',
                        help='the path to the musima classes definition xml file')
    parser.add_argument('--save_dir', default='MUSCIMA++/datasets_r_staff_20_crop', help='The output directory')
    parser.add_argument('--seed', default=314, help='random seed')
    parser.add_argument('--crop_times', default=14, help='number of crops for each image')
                        
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

    print("Annotations Loaded.")

    # clsname2id = {}
    # tree = ElementTree.parse(args.classes)
    # root = tree.getroot()
    # for nodeclass in root:
    #     clsname = nodeclass.find('Name').text
    #     idx = nodeclass.find('Id').text
    #     clsname2id[clsname] = idx

    clsname2id = CLASSNAME2ID

    print("Train...")
    generate(train_docs, clsname2id, args.save_dir, "train", args.crop_times)
    print("Val...")
    generate(val_docs, clsname2id, args.save_dir, "valid", args.crop_times)
    print("Test...")
    generate(test_docs, clsname2id, args.save_dir,  "test", args.crop_times)

    
    print("DONE.")

  