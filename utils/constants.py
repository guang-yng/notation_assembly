from mung.io import parse_node_classes
import os
import json

__all__ = ["node_classes_dict", "node_class_dict_count100", "RESTRICTEDCLASSES20", "ESSENTIALCLASSES"]

node_classes_path = "resources/mff-muscima-mlclasses-annot.xml"
filepath = "resources/mff-muscima-mlclasses-annot.deprules"

class_distribution = "Yolo Result Analysis - Train Res.csv"

node_classes = parse_node_classes(node_classes_path)
node_classes_dict = {node_class.name : node_class.class_id for idx, node_class in enumerate(node_classes)}

with open("utils/train_class_dist.json", 'r') as f:
    cls_count = json.load(f)
node_class_dict_count100 = {name : id for name, id in node_classes_dict.items() 
                            if name in cls_count and cls_count[name] >= 100}

# 20 Restricted classes
RESTRICTEDCLASSES20 = {
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

# Essential classes
ESSENTIALCLSSES_LIST = [
    "noteheadFull",
    "stem",
    "beam",
    "augmentationDot",
    "accidentalSharp",
    "accidentalFlat",
    "accidentalNatural",
    "accidentalDoubleSharp",
    "accidentalDoubleFlat",
    "restWhole",
     "restHalf",
     "restQuarter",
     "rest8th",
     "rest16th",
     "multiMeasureRest",
     "repeat1Bar",
     "legerLine",
     "graceNoteAcciaccatura",
     "noteheadFullSmall",
     "brace",
     "staffGrouping",
     "barline",
     "barlineHeavy",
     "measureSeparator",
     "repeat",
     "repeatDot",
     "articulationStaccato",
     "articulationTenuto",
     "articulationAccent",
     "slur",
     "tie",
     "dynamicCrescendoHairpin",
     "dynamicDiminuendoHairpin",
     "ornament",
     "wiggleTrill",
     "ornamentTrill",
     "arpeggio",
     "glissando",
     "tupleBracket",
     "tuple",
     "gClef",
     "fClef",
     "cClef",
     "keySignature",
     "timeSignature",
     "dynamicsText",
     "tempoText",
     "otherText",
     "numeral0",
     "numeral1",
     "numeral2",
     "numeral3",
     "numeral4",
     "numeral5",
     "numeral6",
     "numeral7",
     "numeral8",
     "otherNumericSign",
     "unclassified",
     "horizontalSpanner",
     "breathMark",
     "noteheadHalf",
     "noteheadWhole",
     "flag8thUp",
     "flag8thDown",
     "flag16thUp",
     "flag16thDown",
     "fermataAbove",
     "fermataBelow",
     "dynamicLetterP",
     "dynamicLetterM",
     "dynamicLetterF",
     "dynamicLetterS",
]
ESSENTIALCLSSES = {clsname : idx for idx, clsname in enumerate(ESSENTIALCLSSES_LIST)}