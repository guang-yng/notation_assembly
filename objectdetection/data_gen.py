import os
import yaml
import shutil
from tqdm import tqdm
import random
from argparse import ArgumentParser
from xml.etree import ElementTree
from mung.io import read_nodes_from_file
import shutil
from PIL import Image
from constants import RESTRICTEDCLASSES20, ESSENTIALCLSSES

def load_split(split_file):
    with open(split_file, 'rb') as hdl:
        split = yaml.load(hdl, Loader=yaml.FullLoader)
    return split

def generate(docs, clsname2id, save_dir, mode, crop_times, image_source_dir):
    images_dir = os.path.join(save_dir, mode, 'images')
    labels_dir = os.path.join(save_dir, mode, 'labels')
    # Create (or clear) directories
    for directory in [images_dir, labels_dir]:
        os.makedirs(directory, exist_ok=True)  # Creates the directory if it doesn't exist, does nothing otherwise

    for doc in tqdm(docs, desc=f'Generating {mode}'):
        doc_name = doc[0].document
        src_path = os.path.join(image_source_dir, f"{doc_name}.png")
        if not os.path.exists(src_path):
            doc_name = doc_name.replace("ideal", "symbol")
            src_path = os.path.join(image_source_dir, f"{doc_name}.png")
        
        if crop_times == 0 or mode == 'test':
            dst_path = os.path.join(images_dir, f"{doc_name}.png")
            
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

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('-d', '--data', 
                        default='MUSCIMA++/v2.0/data/annotations',
                        help="data directory of annotations")
    parser.add_argument('--image_dir', default="MUSCIMA++/datasets_r_staff/images", help='data directory of images')
    parser.add_argument('--classes', default='MUSCIMA++/v2.0/specifications/mff-muscima-mlclasses-annot.xml',
                        help="The path to the musima classes definition xml file. If set to '20', 20 restricted classes are used. If set to 'essential', essential classes are used.")
    parser.add_argument('--save_dir', default='MUSCIMA++/datasets_r_staff_20_crop', help='The output directory')
    parser.add_argument('--save_config', default='data_staff_removed_20_crop.yaml', help='The path to save yaml file')
    parser.add_argument('--split_file', default='../splits/mob_split.yaml', help='The split yaml file.')
    parser.add_argument('--crop_times', default=14, type=int, help='number of crops for each image')
                        
    args = parser.parse_args()

    print("Reading annotations...")
    split_file = load_split(args.split_file)
    docs = {}
    for mode in ("train", "valid", "test"):
        cropobject_fnames = [os.path.join(args.data, f) for f in os.listdir(args.data) if os.path.splitext(os.path.basename(f))[0] in split_file[mode]]
        docs[mode] = [read_nodes_from_file(f) for f in cropobject_fnames]
    print("Annotations Loaded.")

    if args.classes == '20':
        clsname2id = RESTRICTEDCLASSES20
        idclsname = [(0, 'noteheadEmpty')] + [(clsname2id[clsname], clsname) for clsname in clsname2id if clsname2id[clsname] != 0]
    elif args.classes == 'essential':
        clsname2id = ESSENTIALCLSSES
        idclsname = [(clsname2id[clsname], clsname) for clsname in clsname2id]
    else:
        clsname2id = {}
        idclsname = []
        tree = ElementTree.parse(args.classes)
        root = tree.getroot()
        for nodeclass in root:
            clsname = nodeclass.find('Name').text
            idx = nodeclass.find('Id').text
            idclsname.append((int(idx), clsname))
        idclsname.sort()
        for idx, idcls in enumerate(idclsname):
            clsname2id[idcls[1]] = idx

    for mode in "train", "valid", "test":
        print(f"Processing {mode}...")
        generate(docs[mode], clsname2id, args.save_dir, mode, args.crop_times, args.image_dir)
        print("DONE.")

    print("Writing yaml...")
    with open(args.save_config, 'w') as f:
        f.write(f"path: ../{args.save_dir} # dataset root dir\n")
        f.write("train: train/images \nval: valid/images \ntest: test/images \n\n")
        f.write("# Classes\nnames:\n")
        for idx, idcls in enumerate(idclsname):
            f.write(f"  {idx} : {idcls[1]}\n")
    
    print("DONE.")
