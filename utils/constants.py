from mung.io import parse_node_classes
import os
import json

__all__ = ["node_classes_dict", "node_class_dict_count100"]

node_classes_path = "resources/mff-muscima-mlclasses-annot.xml"
filepath = "resources/mff-muscima-mlclasses-annot.deprules"

class_distribution = "Yolo Result Analysis - Train Res.csv"

node_classes = parse_node_classes(node_classes_path)
node_classes_dict = {node_class.name : node_class.class_id for idx, node_class in enumerate(node_classes)}

with open("utils/train_class_dist.json", 'r') as f:
    cls_count = json.load(f)
node_class_dict_count100 = {name : id for name, id in node_classes_dict.items() 
                            if name in cls_count and cls_count[name] >= 100}