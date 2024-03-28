import os, math
import torch, torchvision
import scipy
import numpy as np
from xml.etree import ElementTree
from mung.io import read_nodes_from_file
import xml.etree.ElementTree as ET
from tqdm import tqdm
from PIL import Image, ImageDraw
from model import YOLOSoft
from argparse import ArgumentParser
from ultralytics.utils.plotting import Colors
from constants import RESTRICTEDCLASSES20, ESSENTIALCLSSES, node_classes_dict

PATCH_SIZE=1216
N_CLASSES=128
MARGIN=128
STEP_SIZE=PATCH_SIZE-2*MARGIN

class DashedImageDraw(ImageDraw.ImageDraw):

    def thick_line(self, xy, direction, fill=None, width=0):
        #xy – Sequence of 2-tuples like [(x, y), (x, y), ...]
        #direction – Sequence of 2-tuples like [(x, y), (x, y), ...]
        if xy[0] != xy[1]:
            self.line(xy, fill = fill, width = width)
        else:
            x1, y1 = xy[0]            
            dx1, dy1 = direction[0]
            dx2, dy2 = direction[1]
            if dy2 - dy1 < 0:
                x1 -= 1
            if dx2 - dx1 < 0:
                y1 -= 1
            if dy2 - dy1 != 0:
                if dx2 - dx1 != 0:
                    k = - (dx2 - dx1)/(dy2 - dy1)
                    a = 1/math.sqrt(1 + k**2)
                    b = (width*a - 1) /2
                else:
                    k = 0
                    b = (width - 1)/2
                x3 = x1 - math.floor(b)
                y3 = y1 - int(k*b)
                x4 = x1 + math.ceil(b)
                y4 = y1 + int(k*b)
            else:
                x3 = x1
                y3 = y1 - math.floor((width - 1)/2)
                x4 = x1
                y4 = y1 + math.ceil((width - 1)/2)
            self.line([(x3, y3), (x4, y4)], fill = fill, width = 1)
        return   
        
    def dashed_line(self, xy, dash=(2,2), fill=None, width=0):
        #xy – Sequence of 2-tuples like [(x, y), (x, y), ...]
        for i in range(len(xy) - 1):
            x1, y1 = xy[i]
            x2, y2 = xy[i + 1]
            x_length = x2 - x1
            y_length = y2 - y1
            length = math.sqrt(x_length**2 + y_length**2)
            dash_enabled = True
            postion = 0
            while postion <= length:
                for dash_step in dash:
                    if postion > length:
                        break
                    if dash_enabled:
                        start = postion/length
                        end = min((postion + dash_step - 1) / length, 1)
                        self.thick_line([(round(x1 + start*x_length),
                                          round(y1 + start*y_length)),
                                         (round(x1 + end*x_length),
                                          round(y1 + end*y_length))],
                                        xy, fill, width)
                    dash_enabled = not dash_enabled
                    postion += dash_step
        return

    def dashed_rectangle(self, xy, dash=(2,2), outline=None, width=0):
        #xy - Sequence of [(x1, y1), (x2, y2)] where (x1, y1) is top left corner and (x2, y2) is bottom right corner
        x1, y1 = xy[0]
        x2, y2 = xy[1]
        halfwidth1 = math.floor((width - 1)/2)
        halfwidth2 = math.ceil((width - 1)/2)
        min_dash_gap = min(dash[1::2])
        end_change1 = halfwidth1 + min_dash_gap + 1
        end_change2 = halfwidth2 + min_dash_gap + 1
        odd_width_change = (width - 1)%2        
        self.dashed_line([(x1 - halfwidth1, y1), (x2 - end_change1, y1)],
                         dash, outline, width)       
        self.dashed_line([(x2, y1 - halfwidth1), (x2, y2 - end_change1)],
                         dash, outline, width)
        self.dashed_line([(x2 + halfwidth2, y2 + odd_width_change),
                          (x1 + end_change2, y2 + odd_width_change)],
                         dash, outline, width)
        self.dashed_line([(x1 + odd_width_change, y2 + halfwidth2),
                          (x1 + odd_width_change, y1 + end_change2)],
                         dash, outline, width)
        return

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
    parser.add_argument('--data', default='MUSCIMA++/v2.0', help='The dataset path. Used to link to original images and read ground truths.')
    parser.add_argument('--images', default='MUSCIMA++/datasets_r_staff/images', help='The path to images to be predict.')
    parser.add_argument('--classes', default='MUSCIMA++/v2.0/specifications/mff-muscima-mlclasses-annot.xml',
                        help="The path to the musima classes definition xml file. If set to '20', 20 restricted classes are used. If set to 'essential', essential classes are used.")
    parser.add_argument('--visualize', action='store_true', help='Whether visualize the result')
    parser.add_argument('--grids', action='store_true', help='Whether to visualize the girds. Only valid when --visualize is set.')
    parser.add_argument('--links', action='store_true', help='Whether to generate psuedo edges in annotations.')
    parser.add_argument('--batch_size', default=16, help='The batch size for inference.')
    parser.add_argument('--save_dir', default='MUSCIMA++/v2.0_gen', help='The directory to save results')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'data'), exist_ok=True)
    os.chmod(args.save_dir, 0o777)
    os.chmod(os.path.join(args.save_dir, 'data'), 0o777)
    save_img_dir = os.path.join(args.save_dir, 'data', 'images')
    if not os.path.exists(save_img_dir):
        os.symlink(os.path.abspath(os.path.join(args.data, 'data', 'images')), save_img_dir, target_is_directory=True)

    print("Loading model...")
    model = YOLOSoft(args.model)
    print("Done.")
    images = []
    indices = []
    for img_name in tqdm(os.listdir(args.images), desc='splitting images..'):
        img = Image.open(os.path.join(args.images, img_name))
        W, H = img.size
        x_steps, y_steps = (W-1)//STEP_SIZE + 1, (H-1)//STEP_SIZE + 1
        for x_id in range(x_steps):
            for y_id in range(y_steps):
                offset = (x_id*STEP_SIZE-MARGIN, y_id*STEP_SIZE-MARGIN)
                subimage = img.crop((offset[0], offset[1], offset[0]+PATCH_SIZE, offset[1]+PATCH_SIZE))
                images.append(subimage)
                indices.append((img_name.replace('symbol', 'ideal'), x_id, y_id))
    results = []
    for idx in tqdm(range(0, len(images), args.batch_size), desc='predicting...'):
        batch_imgs = images[idx:min(idx+args.batch_size, len(images))]
        results.extend(model(batch_imgs, verbose=False))
    imgname2preds = {}
    for index, result in tqdm(zip(indices, results), desc="merging results...", total=len(indices)):
        img_name, x_id, y_id = index
        offset = (x_id*STEP_SIZE-MARGIN, y_id*STEP_SIZE-MARGIN)
        if img_name not in imgname2preds:
            imgname2preds[img_name] = {}
        preds_dict = imgname2preds[img_name]
        preds_list = []
        preds_dict[(x_id, y_id)] = preds_list

        def remove_truncated(plist, true_box, prob):
            to_remove = None
            for idx, (box_pre, prob_pre) in enumerate(plist):
                if bbox_iop(box_pre, true_box)*np.dot(prob.data, prob_pre.data) > 0.20:
                    to_remove = idx
                    break
            if to_remove:
                del plist[to_remove]

        for box, prob in zip(result.boxes, result.probs):
            bbox = box.xyxy.squeeze().cpu().numpy()
            probdata = prob.cpu().numpy()
            left, top, right, bottom = bbox
            if (right <= 2*MARGIN and x_id > 0) or (bottom <= 2*MARGIN and y_id > 0):
                continue
            true_box = bbox + np.array([offset[0], offset[1], offset[0], offset[1]])
            if (x_id > 0 and left < 2*MARGIN):
                assert (x_id-1, y_id) in preds_dict
                remove_truncated(preds_dict[(x_id-1, y_id)], true_box, probdata)
            if (y_id > 0 and top < 2*MARGIN):
                assert (x_id, y_id-1) in preds_dict
                remove_truncated(preds_dict[(x_id, y_id-1)], true_box, probdata)
            preds_list.append((true_box, probdata))
            
    # Visualize
    if args.visualize:
        os.makedirs(os.path.join(args.save_dir, 'data', 'visualization'), exist_ok=True)
        os.chmod(os.path.join(args.save_dir, 'data', 'visualization'), 0o777)
        color_palette = Colors()
        for img_name, preds_dict in tqdm(imgname2preds.items(), desc="saving visualization files..."):
            img_n = img_name.replace('ideal', 'symbol')
            img = Image.open(os.path.join(args.images, img_n)).convert("RGB")
            draw = DashedImageDraw(img)
            for (x_id, y_id), preds_list in preds_dict.items():
                for box, prob in preds_list:
                    draw.rectangle(tuple(box), outline=color_palette(int(prob.top1)), width=3)
                if args.grids:
                    draw.rectangle((x_id*STEP_SIZE, y_id*STEP_SIZE, (x_id+1)*STEP_SIZE, (y_id+1)*STEP_SIZE), outline=(255, 255, 255), width=2)
                    draw.dashed_rectangle(((x_id*STEP_SIZE-MARGIN, y_id*STEP_SIZE-MARGIN), ((x_id+1)*STEP_SIZE+MARGIN, (y_id+1)*STEP_SIZE+MARGIN)), dash=(10, 10), outline=(255, 255, 255), width=2)
            img.save(os.path.join(args.save_dir, 'data', 'visualization', img_n))
    
    # Load classes
    clsname2id = node_classes_dict
    id2clsname = {idx : clsname for clsname, idx in clsname2id.items()}
    if args.classes == '20':
        clsnames = RESTRICTEDCLASSES20.keys()
        clsname2id["noteheadWhole"] = clsname2id["noteheadHalf"]
        id2clsname[clsname2id["noteheadHalf"]] = "noteheadHalf"
    elif args.classes == 'essential':
        clsnames = ESSENTIALCLSSES.keys()
    else:
        clsnames = clsname2id.keys()

    # Generate pseudo links
    if args.links:
        imgname2links, imgname2unlinks = {}, {}
        for img_name, preds_dict in tqdm(imgname2preds.items(), desc="creating pseudo links..."):
            pseudo_links, pseudo_unlinks = [], []
            imgname2links[img_name] = pseudo_links
            imgname2unlinks[img_name] = pseudo_unlinks
            boxes, probs = [], []
            for preds_list in preds_dict.values():
                for box, prob in preds_list:
                    boxes.append(box)
                    probs.append(prob.data)
                    pseudo_links.append([])
                    pseudo_unlinks.append([])
            boxes = np.stack(boxes)
            probs = np.stack(probs)
            doc_name = img_name.split('.')[0]
            nodes = read_nodes_from_file(os.path.join(args.data, 'data', 'annotations', f"{doc_name}.xml"))
            nodes = [node for node in nodes if node.class_name in clsnames]
            boxes_g = []
            prob_matrix = []
            id2idx = {}
            for idx, node in enumerate(nodes):
                id2idx[node.id] = idx
                prob_matrix.append(probs[:, clsname2id[node.class_name]])
                boxes_g.append((node.left, node.top, node.right, node.bottom))
            prob_matrix = np.stack(prob_matrix).transpose()
            box_matrix = torchvision.ops.box_iou(torch.tensor(boxes), torch.tensor(boxes_g)).numpy()
            cost_matrix = np.multiply(box_matrix, prob_matrix)
            row_indices, col_indices = scipy.optimize.linear_sum_assignment(-cost_matrix)
            match_b = {col_idx:row_idx for row_idx, col_idx in zip(row_indices, col_indices) if cost_matrix[row_idx, col_idx] > 0.05}
            unconnected_idx = set((i, j) for i in match_b for j in match_b if (i <= j))
            for col_idx in match_b:
                for to_id in nodes[col_idx].outlinks:
                    if to_id not in id2idx:
                        continue
                    if id2idx[to_id] in match_b:
                        unconnected_idx.remove((col_idx, id2idx[to_id]) if col_idx < id2idx[to_id] else (id2idx[to_id], col_idx) )
                        pseudo_links[match_b[col_idx]].append(match_b[id2idx[to_id]])
            for idx1, idx2 in unconnected_idx:
                pseudo_unlinks[match_b[idx1]].append(match_b[idx2])

    # Saving results
    os.makedirs(os.path.join(args.save_dir, 'data', 'annotations'), exist_ok=True)
    os.chmod(os.path.join(args.save_dir, 'data', 'annotations'), 0o777)
    for img_name, preds_dict in tqdm(imgname2preds.items(), desc="saving predictions..."):
        doc_name = img_name.split('.')[0].replace('symbol', 'ideal')
        root = ET.Element("Nodes")
        root.set('dataset', "MUSCIMA-pp_2.0")
        root.set('document', doc_name)
        root.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
        root.set("xsi:noNamespaceSchemaLocation", "CVC-MUSCIMA_Schema.xsd")
        count = 0
        soft_classes = []
        for preds_list in preds_dict.values():
            for box, prob in preds_list:
                node = ET.SubElement(root, "Node")
                soft_classes.append(prob.data)
                ET.SubElement(node, "Id").text = str(count)
                ET.SubElement(node, "ClassName").text = id2clsname[int(prob.top1)]
                ET.SubElement(node, "Top").text = str(round(box[1].item()))
                ET.SubElement(node, "Left").text = str(round(box[0].item()))
                ET.SubElement(node, "Width").text = str(round((box[2]-box[0]).item()))
                ET.SubElement(node, "Height").text = str(round((box[3]-box[1]).item()))
                if args.links:
                    ET.SubElement(node, "Outlinks").text = " ".join(str(i) for i in imgname2links[img_name][count])
                    data = ET.SubElement(node, "Data")
                    ET.SubElement(data, "DataItem", {"key": "unoutlinks", "type": "list[int]"}).text = " ".join(str(i) for i in imgname2unlinks[img_name][count])
                count += 1
        ET.indent(root, space="\t", level=0)
        ET.ElementTree(root).write(os.path.join(args.save_dir, 'data', 'annotations', f"{doc_name}.xml"))
        np.save(os.path.join(args.save_dir, 'data', 'annotations', f"{doc_name}.npy"), soft_classes)
