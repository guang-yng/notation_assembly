import os
from tqdm import tqdm
from argparse import ArgumentParser
from xml.etree import ElementTree
from mung.io import read_nodes_from_file

from PIL import Image
import os
from tqdm import tqdm

def visualize(docs, images, clsname2id, save_dir):
    images_dir = os.path.join(save_dir)
    os.makedirs(images_dir, exist_ok=True)
    clsname2img = {}

    for doc, image in tqdm(zip(docs, images), desc=f'Iterating all docs'):
        for node in doc:
            if node.class_name in clsname2img:
                continue
            _img = image.crop((node.left, node.top, node.right, node.bottom))
            clsname2img[node.class_name] = _img
    for clsname in tqdm(clsname2img, desc="Saving..."):
        save_path = os.path.join(save_dir, f"{clsname2id[clsname]}_{clsname}.png")
        clsname2img[clsname].save(save_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-d', '--data', 
                        default='data/MUSCIMA++/v2.0/data/annotations',
                        help="data directory")
    parser.add_argument('--classes', default='data/MUSCIMA++/v2.0/specifications/mff-muscima-mlclasses-annot.xml',
                        help='the path to the musima classes definition xml file')
    parser.add_argument('--save_dir', default='class_sample_images', help='The output directory')
                        
    args = parser.parse_args()

    cropobject_fnames = [os.path.join(args.data, f) for f in os.listdir(args.data) if f.endswith('xml')]
    docs = [read_nodes_from_file(f) for f in cropobject_fnames]
    data_prefix, annotation_folder = os.path.split(args.data)
    image_dir = os.path.join(data_prefix, 'images', annotation_folder)
    images = [Image.open(os.path.join(image_dir, f"{doc[0].document}.png")) for doc in docs]

    print("Annotations & Images Loaded.")

    clsname2id = {}
    tree = ElementTree.parse(args.classes)
    root = tree.getroot()
    for nodeclass in root:
        clsname = nodeclass.find('Name').text
        idx = nodeclass.find('Id').text
        clsname2id[clsname] = idx

    visualize(docs, images, clsname2id, args.save_dir)
