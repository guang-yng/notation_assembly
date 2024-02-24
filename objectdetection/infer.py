import os
import numpy as np
from xml.etree import ElementTree
import xml.etree.cElementTree as ET
from tqdm import tqdm
from PIL import Image, ImageDraw
from ultralytics import YOLO
from argparse import ArgumentParser
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.plotting import Colors
from constants import RESTRICTEDCLASSES20, ESSENTIALCLSSES

PATCH_SIZE=1216
N_CLASSES=128
MARGIN=128
STEP_SIZE=PATCH_SIZE-2*MARGIN

def bbox_iop(box1, box2, eps=1e-7):
    b1_x1, b1_y1, b1_x2, b1_y2 = box1
    b2_x1, b2_y1, b2_x2, b2_y2 = box2
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps

    # Intersection area
    inter = max(min(b1_x2, b2_x2) - max(b1_x1, b2_x1), 0) * max(
        min(b1_y2, b2_y2) - max(b1_y1, b2_y1), 0
    )

    return inter / (w1*h1)
    

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--model', default='runs/detect/train-v8l-b8-i640-essn/train/weights/best.pt', help='The model to load.')
    parser.add_argument('--data', default='MUSCIMA++/datasets_r_staff_essential_crop', help='The dataset path.')
    parser.add_argument('--classes', default='MUSCIMA++/v2.0/specifications/mff-muscima-mlclasses-annot.xml',
                        help="The path to the musima classes definition xml file. If set to '20', 20 restricted classes are used. If set to 'essential', essential classes are used.")
    parser.add_argument('--visualize', action='store_true', help='Whether visualize the result')
    parser.add_argument('--grids', action='store_true', help='Whether to visualize the girds. Only valid when --visualize is set.')
    parser.add_argument('--batch_size', default=8, help='The batch size for inference.')
    parser.add_argument('--save_dir', default='predict', help='The directory to save results')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    print("Loading model...")
    model = YOLO(args.model)
    print("Done.")
    test_path = os.path.join(args.data, 'test', 'images')
    images = []
    indices = []
    for img_name in tqdm(os.listdir(test_path), desc='splitting images..'):
        img = Image.open(os.path.join(test_path, img_name))
        W, H = img.size
        x_steps, y_steps = (W-1)//STEP_SIZE + 1, (H-1)//STEP_SIZE + 1
        for x_id in range(x_steps):
            for y_id in range(y_steps):
                offset = (x_id*STEP_SIZE-MARGIN, y_id*STEP_SIZE-MARGIN)
                subimage = img.crop((offset[0], offset[1], offset[0]+PATCH_SIZE, offset[1]+PATCH_SIZE))
                images.append(subimage)
                indices.append((img_name, x_id, y_id))
    results = []
    for idx in tqdm(range(0, len(images), args.batch_size), desc='predicting...'):
        batch_imgs = images[idx:min(idx+args.batch_size, len(images))]
        results.extend(model(batch_imgs))
    imgname2preds = {}
    for index, result in tqdm(zip(indices, results), desc="merging results...", total=len(indices)):
        img_name, x_id, y_id = index
        offset = (x_id*STEP_SIZE-MARGIN, y_id*STEP_SIZE-MARGIN)
        if img_name not in imgname2preds:
            imgname2preds[img_name] = {}
        preds_dict = imgname2preds[img_name]
        preds_list = []
        preds_dict[(x_id, y_id)] = preds_list

        def remove_truncated(plist, true_box, clsid):
            to_remove = None
            for idx, (box_pre, cls_pre) in enumerate(plist):
                if cls_pre != clsid:
                    continue
                if bbox_iop(box_pre, true_box) > 0.80:
                    to_remove = idx
                    break
            if to_remove:
                del plist[to_remove]

        for box in result.boxes:
            bbox = box.xyxy.squeeze().cpu().numpy()
            left, top, right, bottom = bbox
            if (right <= 2*MARGIN and x_id > 0) or (bottom <= 2*MARGIN and y_id > 0):
                continue
            true_box = bbox + np.array([offset[0], offset[1], offset[0], offset[1]])
            clsid = int(box.cls.item())
            if (x_id > 0 and left < 2*MARGIN):
                assert (x_id-1, y_id) in preds_dict
                remove_truncated(preds_dict[(x_id-1, y_id)], true_box, clsid)
            if (y_id > 0 and top < 2*MARGIN):
                assert (x_id, y_id-1) in preds_dict
                remove_truncated(preds_dict[(x_id, y_id-1)], true_box, clsid)
            preds_list.append((true_box, clsid))
            
    # Visualize
    if args.visualize:
        os.makedirs(os.path.join(args.save_dir, 'visualization'), exist_ok=True)
        color_palette = Colors()
        for img_name, preds_dict in tqdm(imgname2preds.items(), desc="saving visualization files..."):
            img = Image.open(os.path.join(test_path, img_name)).convert("RGB")
            draw = ImageDraw.Draw(img)
            for (x_id, y_id), preds_list in preds_dict.items():
                for box, clsid in preds_list:
                    draw.rectangle(tuple(box), outline=color_palette(clsid), width=3)
                x_id*STEP_SIZE
                if args.grids:
                    draw.rectangle((x_id*STEP_SIZE, y_id*STEP_SIZE, (x_id+1)*STEP_SIZE, (y_id+1)*STEP_SIZE), outline=(255, 255, 255), width=2)
                    draw.rectangle((x_id*STEP_SIZE+MARGIN, y_id*STEP_SIZE+MARGIN, (x_id+1)*STEP_SIZE-MARGIN, (y_id+1)*STEP_SIZE-MARGIN), outline=(255, 255, 255), width=2)
            img.save(os.path.join(args.save_dir, 'visualization', img_name))
    
    # Load classes
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
    id2clsname = {clsname2id[k]: k for k in clsname2id}

    # Saving results
    os.makedirs(os.path.join(args.save_dir, 'annotations'), exist_ok=True)
    for img_name, preds_dict in tqdm(imgname2preds.items(), desc="saving predictions..."):
        doc_name = img_name.split('.')[0].replace('symbol', 'ideal')
        root = ET.Element("Nodes")
        root.set('dataset', "MUSCIMA-pp_2.0")
        root.set('document', doc_name)
        root.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
        root.set("xsi:noNamespaceSchemaLocation", "CVC-MUSCIMA_Schema.xsd")
        count = 0
        for preds_list in preds_dict.values():
            for box, clsid in preds_list:
                node = ET.SubElement(root, "Node")
                ET.SubElement(node, "Id").text = str(count)
                ET.SubElement(node, "ClassName").text = id2clsname[clsid]
                ET.SubElement(node, "Top").text = str(box[1].item())
                ET.SubElement(node, "Left").text = str(box[0].item())
                ET.SubElement(node, "Width").text = str((box[2]-box[0]).item())
                ET.SubElement(node, "Height").text = str((box[3]-box[1]).item())
                count += 1
        ET.ElementTree(root).write(os.path.join(args.save_dir, 'annotations', f"{doc_name}.xml"))
