from mung.io import parse_node_classes
import os

__all__ = ["node_classes_dict"]

node_classes_path = "resources/mff-muscima-mlclasses-annot.xml"
filepath = "resources/mff-muscima-mlclasses-annot.deprules"

node_classes = parse_node_classes(node_classes_path)
node_classes_dict = {node_class.name : node_class.class_id for idx, node_class in enumerate(node_classes)}

