import copy
import os
from glob import glob
from typing import List, Tuple, Dict
import random

import numpy as np
import yaml
from PIL import Image
from mung.grammar import DependencyGrammar
from mung.graph import NotationGraph
from mung.io import read_nodes_from_file
from mung.node import Node, bounding_box_intersection

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .utils import config2data_pool_dict
from utils.constants import node_classes_dict, node_class_dict_count100


class MunglinkerDataError(ValueError):
    pass


class PairwiseMungoDataPool(Dataset):
    """This class implements the basic data pool for munglinker experiments
    that outputs just pairs of MuNG nodes from the same document. Using this
    pool means that your preparation function will have to deal with everything
    else, like having at its disposal also the appropriate image from which
    to get the input patch, if need be in your model.

    It is entirely sufficient for training the baseline decision trees without
    complications, though.
    """

    def __init__(self, mungs: List[NotationGraph],
                 images: List[np.ndarray],
                 max_edge_length: int,
                 max_negative_samples: int,
                 patch_size: Tuple[int, int],
                 zoom: float,
                 grammar: DependencyGrammar = None,
                 filter_pairs: bool = True,
                 normalize_bbox: bool = True,
                 class_perturb=0.0):
        """Initialize the data pool.

        :param mungs: The NotationGraph objects for each document
            in the dataset.

        :param images: The corresponding images of the MuNGs. If
            not provided, binary masks will be generated as a union
            of all the MuNGos' masks.

        :param max_edge_length: The longest allowed edge length, measured
            as the minimum distance of the bounding boxes of the mungo
            pair in question.

        :param max_negative_samples: The maximum number of mungos sampled
            as negative examples per mungo.

        :param patch_size: What the size of the extracted patch should
            be (after applying zoom), specified as ``(rows, columns)``.

        :param zoom: The rescaling factor. Setting this to 0.5 means
            the image will be downscaled to half the height & width
            before the patch is extracted.

        :param filter_pairs: whether or not to filter the pairs

        :param normalize_bbox: whether or not to normalize the bounding box

        """
        # self.filter_by_distance_portion = []
        self.mungs = mungs
        self.images = images

        self.normalize_bbox = normalize_bbox
        if normalize_bbox:
            for mung, image in zip(self.mungs, self.images):
                for node in mung.vertices:
                    node.__top = node.top / image.shape[0]
                    node.__left = node.left / image.shape[1]
                    node.__bottom = node.bottom / image.shape[0]
                    node.__right = node.right / image.shape[1]

        self.max_edge_length = max_edge_length
        self.max_negative_samples = max_negative_samples

        self.patch_size = patch_size
        self.patch_height = patch_size[0]
        self.patch_width = patch_size[1]

        self.zoom = zoom
        if self.zoom != 1.0:
            self.__zoom_images()
            self.__zoom_mungs()

        self.grammar = grammar

        self.length = 0
        self.filter_pairs = filter_pairs
        self.class_perturb = class_perturb
        
        self.prepare_train_entities()
        # print(np.mean(self.filter_by_distance_portion))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        mung_from = self.all_mungo_pairs[idx][0]
        mung_to = self.all_mungo_pairs[idx][1]
        if self.normalize_bbox:
            image_shape = self.images[self.train_entities[idx][0]].shape
            reshape_weight = torch.tensor([1 / image_shape[0], 1 / image_shape[1],
                                           1 / image_shape[0], 1 / image_shape[1]])
        else:
            reshape_weight = torch.ones(4)

        source_bbox = torch.tensor(mung_from.bounding_box) * reshape_weight
        if random.random() < self.class_perturb:
            source_class = torch.tensor(random.choice(list(node_class_dict_count100.values())))
        else:
            source_class = torch.tensor(node_classes_dict[mung_from.class_name])
        target_bbox = torch.tensor(mung_to.bounding_box) * reshape_weight
        target_class = torch.tensor(node_classes_dict[mung_to.class_name])
        label = torch.tensor(mung_to.id in mung_from.outlinks).unsqueeze(-1).float()

        return dict(
            source_bbox=source_bbox,
            source_class=source_class,
            target_bbox=target_bbox,
            target_class=target_class,
            label=label
        )

    def __zoom_images(self):
        images_zoomed = []
        import cv2
        for image in self.images:
            img_copy = image * 1.0
            img_zoomed = cv2.resize(img_copy, dsize=None,
                                    fx=self.zoom, fy=self.zoom).astype(image.dtype)
            images_zoomed.append(img_zoomed)
        self.images = images_zoomed

    def __zoom_mungs(self):
        if self.zoom is None:
            return
        if self.zoom == 1.0:
            return
        for mung in self.mungs:
            for m in mung.vertices:
                m.scale(zoom=self.zoom)


    def prepare_train_entities(self):
        """Extract the triplets.
        Extract MuNGo list that the train_entities will then refer to.

        The triplets will be represented as ``(i_image, m_from, m_to)``,
        where ``i_image`` points to the originating image and ``m_from``
        and ``m_to`` are one instance of sampled mungo pairs.
        """
        self.train_entities = []
        self.all_mungo_pairs = []  # type: List[Tuple[Node, Node]]
        self.inference_graph = {}
        number_of_samples = 0
        for mung_index, mung in enumerate(tqdm(self.mungs, desc="Loading MuNG-pairs")):
            if self.filter_pairs:
                object_pairs = self.get_all_neighboring_object_pairs(
                    mung.vertices,
                    max_object_distance=self.max_edge_length,
                    grammar=self.grammar)
            else:
                object_pairs = []
                for m_from in mung.vertices:
                    for m_to in mung.vertices:
                        if m_from.id != m_to.id:
                            object_pairs.append((m_from, m_to))
            self.inference_graph[mung_index] = object_pairs
            for (m_from, m_to) in object_pairs:
                self.all_mungo_pairs.append((m_from, m_to))
                self.train_entities.append([mung_index, number_of_samples])
                number_of_samples += 1

        self.length = number_of_samples

    
    def get_inference_graph(self):
        # Convert everything into a dict of batched tensor for each graph
        print("Preparing graph for inference...")
        for idx, graph in tqdm(self.inference_graph.items()):
            tensor_dict = {"source_bbox": [], "source_class": [], "target_bbox": [], "target_class": [], "label": [], 
                           "source_id": [], "target_id": []}
            for pair in graph:
                source_bbox = torch.tensor(pair[0].bounding_box)
                source_class = torch.tensor(node_classes_dict[pair[0].class_name])
                target_bbox = torch.tensor(pair[1].bounding_box)
                target_class = torch.tensor(node_classes_dict[pair[1].class_name])
                label = torch.tensor(pair[1].id in pair[0].outlinks).unsqueeze(-1).float()
                tensor_dict["source_bbox"].append(source_bbox)
                tensor_dict["source_class"].append(source_class)
                tensor_dict["target_bbox"].append(target_bbox)
                tensor_dict["target_class"].append(target_class)
                tensor_dict["label"].append(label)
                tensor_dict['source_id'].append(torch.tensor(pair[0].id))
                tensor_dict['target_id'].append(torch.tensor(pair[1].id))
            self.inference_graph[idx] = {k: torch.stack(v) for k, v in tensor_dict.items()}

        return self.inference_graph


    def get_closest_objects(self, nodes: List[Node], threshold) -> Dict[Node, List[Node]]:
        """For each pair of Nodes, compute the closest distance between their
        bounding boxes.

        :returns: A dict of dicts, indexed by objid, then objid, then distance.
        """
        close_objects = {}
        for c in nodes:
            close_objects[c] = []

        for c in nodes:
            for d in nodes:
                distance = c.distance_to(d)
                if distance < threshold:
                    close_objects[c].append(d)
                    close_objects[d].append(c)

        # Remove duplicates from lists
        for key, neighbors in close_objects.items():
            unique_neighbors = list(dict.fromkeys(neighbors))
            close_objects[key] = unique_neighbors

        return close_objects

    def get_all_neighboring_object_pairs(self, nodes: List[Node],
                                         max_object_distance,
                                         grammar=None) -> List[Tuple[Node, Node]]:
        close_neighbors = self.get_closest_objects(nodes, max_object_distance)

        example_pairs_dict = {}
        for c in close_neighbors:
            if grammar is None:
                example_pairs_dict[c] = close_neighbors[c]
            else:
                example_pairs_dict[c] = [d for d in close_neighbors[c] if grammar.validate_edge(c.class_name, d.class_name)]

        examples = []
        for c in example_pairs_dict:
            for d in example_pairs_dict[c]:
                examples.append((c, d))

        # Validate that every link from the ground-truth also has a candidate in the examples - all positives are included
        # print("Validating...")
        # print(f"Max distance: {max_object_distance}")
        # id_to_cropobject_mapping = {cropobject.id: cropobject for cropobject in nodes}
        # total_edges, filter_by_distance = 0, 0
        # for n1 in nodes:
        #     for n2id in n1.outlinks:
        #         n2 = id_to_cropobject_mapping[n2id]
        #         if not grammar.validate_edge(n1.class_name, n2.class_name):
        #             if 'staff' not in n1.class_name and 'staff' not in n2.class_name and 'Text' not in n1.class_name and 'Text' not in n2.class_name and 'dottedHorizontalSpanner' not in n1.class_name:
        #                 print(n1.class_name, n2.class_name)
        #                 exit(0)
        #             continue
        #         total_edges += 1
        #         if n1.distance_to(n2) >= max_object_distance:
        #             filter_by_distance += 1
        # print(f"Filter by distance portion: {filter_by_distance/total_edges}")
        # self.filter_by_distance_portion.append(filter_by_distance/total_edges)

        return examples


##############################################################################

# Technically these methods are the same, but there might in the future
# be different validation checks.

def load_split(split_file):
    with open(split_file, 'rb') as hdl:
        split = yaml.load(hdl, Loader=yaml.FullLoader)
    return split


def load_config(config_file: str):
    with open(config_file, 'rb') as hdl:
        config = yaml.load(hdl, Loader=yaml.FullLoader)
    return config


def __load_mung(filename: str, exclude_classes: List[str]) -> NotationGraph:
    mungos = read_nodes_from_file(filename)
    mung = NotationGraph(mungos)
    objects_to_exclude = [m for m in mungos if m.class_name in exclude_classes]
    for m in objects_to_exclude:
        mung.remove_vertex(m.id)
    return mung


def __load_image(filename: str) -> np.ndarray:
    image = np.array(Image.open(filename).convert('1')).astype('uint8')
    return image


def __load_munglinker_data(mung_root: str, images_root: str,
                           include_names: List[str] = None,
                           max_items: int = None,
                           exclude_classes=None,
                           masks_to_bounding_boxes=False):
    """Loads the MuNGs and corresponding images from the given folders.
    All *.xml files in ``mung_root`` are considered MuNG files, all *.png
    files in ``images_root`` are considered image files.

    Use this to get data for initializing the PairwiseMungoDataPool.

    :param mung_root: Directory containing MuNG XML files.

    :param images_root: Directory containing underlying image files (png).

    :param include_names: Only load files such that their basename is in
        this list. Useful for loading train/test/validate splits.

    :param max_items: Load at most this many files.

    :param exclude_classes: When loading the MuNG, exclude notation objects
        that are labeled as one of these classes. (Most useful for excluding
        staff objects.)

    :param masks_to_bounding_boxes: If set, will replace the masks of the
        loaded MuNGOs with everything in the corresponding bounding box
        of the image. This is to make the training data compatible with
        the runtime outputs of RCNN-based detectors, which only output
        the bounding box, not the mask.

    :returns: mungs, images  -- a tuple of lists.
    """
    if exclude_classes is None:
        exclude_classes = {}

    all_mung_files = glob(mung_root + "/**/*.xml", recursive=True)
    mung_files_in_this_split = sorted([f for f in all_mung_files if os.path.splitext(os.path.basename(f))[0] in include_names])

    all_image_files = glob(images_root + "/**/*.png", recursive=True)
    image_files_in_this_split = sorted([f for f in all_image_files if
                                 os.path.splitext(os.path.basename(f))[0] in include_names])

    mungs = []
    images = []
    for mung_file, image_file in tqdm(zip(mung_files_in_this_split, image_files_in_this_split),
                                      desc="Loading mung/image pairs from disk",
                                      total=len(mung_files_in_this_split)):
        mung = __load_mung(mung_file, exclude_classes)
        mungs.append(mung)

        image = __load_image(image_file)
        images.append(image)

        # This is for training on bounding boxes,
        # which needs to be done in order to then process
        # R-CNN detection outputs with Munglinker trained on ground truth
        if masks_to_bounding_boxes:
            for mungo in mung.vertices:
                t, l, b, r = mungo.bounding_box
                image_mask = image[t:b, l:r]
                mungo.set_mask(image_mask)

        if max_items is not None:
            if len(mungs) >= max_items:
                break

    return mungs, images


def load_munglinker_data(mung_root, images_root, split_file,
                         config_file=None,
                         load_training_data=True,
                         load_validation_data=True,
                         load_test_data=False,
                         exclude_classes=None,
                         class_perturb=0.0) -> Dict[str, PairwiseMungoDataPool]:
    """Loads the train/validation/test data pools for the MuNGLinker
    experiments.

    :param mung_root: Directory containing MuNG XML files.

    :param images_root: Directory containing underlying image files (png).

    :param split_file: YAML file that defines which items are for training,
        validation, and test.

    :param config_file: YAML file defining further experiment properties.
        Not used so far.

    :param load_training_data: Whether or not to load the training data
    :param load_validation_data: Whether or not to load the validation data
    :param load_test_data: Whether or not to load the test data

    :param exclude_classes: When loading the MuNG, exclude notation objects
        that are labeled as one of these classes. (Most useful for excluding
        staff objects.)

    :return: ``dict(train=training_pool, valid=validation_pool, test=test_pool)``
    """
    split = load_split(split_file)
    train_on_bounding_boxes = False

    if config_file is None:
        print("No configuration file found. Terminating")
        exit(-1)

    config = load_config(config_file)
    data_pool_dict = config2data_pool_dict(config)
    data_pool_dict['class_perturb'] = class_perturb

    if 'TRAIN_ON_BOUNDING_BOXES' in config:
        train_on_bounding_boxes = config['TRAIN_ON_BOUNDING_BOXES']

    validation_data_pool_dict = copy.deepcopy(data_pool_dict)
    if 'VALIDATION_MAX_NEGATIVE_EXAMPLES_PER_OBJECT' in config:
        validation_data_pool_dict['max_negative_samples'] = \
            config['VALIDATION_MAX_NEGATIVE_EXAMPLES_PER_OBJECT']

    training_pool = None
    validation_pool = None
    test_pool = None

    if load_training_data:
        print("Loading training data...")
        tr_mungs, tr_images = __load_munglinker_data(mung_root, images_root,
                                                     include_names=split['train'],
                                                     exclude_classes=exclude_classes,
                                                     masks_to_bounding_boxes=train_on_bounding_boxes)
        training_pool = PairwiseMungoDataPool(mungs=tr_mungs, images=tr_images, **data_pool_dict)

    if load_validation_data:
        print("Loading validation data...")
        va_mungs, va_images = __load_munglinker_data(mung_root, images_root,
                                                     include_names=split['valid'],
                                                     exclude_classes=exclude_classes,
                                                     masks_to_bounding_boxes=train_on_bounding_boxes)
        validation_pool = PairwiseMungoDataPool(mungs=va_mungs, images=va_images, filter_pairs=True, **validation_data_pool_dict)

    if load_test_data:
        print("Loading test data...")
        te_mungs, te_images = __load_munglinker_data(mung_root, images_root,
                                                     include_names=split['test'],
                                                     exclude_classes=exclude_classes,
                                                     masks_to_bounding_boxes=train_on_bounding_boxes)
        test_pool = PairwiseMungoDataPool(mungs=te_mungs, images=te_images, filter_pairs=True, **data_pool_dict)

    return dict(train=training_pool, valid=validation_pool, test=test_pool)