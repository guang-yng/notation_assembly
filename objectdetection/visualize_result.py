from PIL import Image
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import numpy as np

MODEL_PATH = '/homes/iws/lq9/notation_assembly/runs/detect/train24-best-1280/weights/best.pt'
def visualize_test_result(image_path):
    # tutorial https://docs.ultralytics.com/modes/predict/#plotting-results
    # Load a pretrained YOLOv8n model
    model = YOLO('/homes/iws/lq9/notation_assembly/runs/detect/train24-best-1280/weights/best.pt')

    results = model(image_path)  # results list

    # Show the results
    for r in results:
        # print(r)
        # print(r.boxes.xywhn) 
        im_array = r.plot(font_size = 10, labels = False)  # plot a BGR numpy array of predictions
        im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
        im.show()  # show image
        im.save('result1280.jpg')  # save image

def get_predictions(image_path):
    # tutorial https://docs.ultralytics.com/modes/predict/#plotting-results
    # Load a pretrained YOLOv8n model
    model = YOLO(MODEL_PATH)

    results = model.predict(image_path, save_txt = True)  # results list
     

def draw_and_save_bounding_box(image_path, annotations, class_labels, output_path):
    """
    Draw bounding boxes with class labels on an image and save the result.
    
    Parameters:
    - image_path: path to the image file.
    - annotations: list of annotations in YOLO format [(class_id, x_center, y_center, width, height), ...].
    - class_labels: dictionary mapping class IDs to class labels.
    - output_path: path to save the image with drawn bounding boxes.
    """
    # Load the image
    image = cv2.imread(image_path)
    h, w, _ = image.shape

    # Generate a unique color for each class ID
    colors = {class_id: np.random.randint(0, 255, size=3).tolist() for class_id in class_labels.keys()}
    colors
    print(colors)
    # Draw each bounding box
    for annotation in annotations:
        class_id, x_center, y_center, width, height = annotation
        class_label = class_labels.get(class_id, 'Unknown')

        # Convert normalized positions to absolute positions
        x_center, y_center, width, height = x_center * w, y_center * h, width * w, height * h
        x_min, y_min = int(x_center - width / 2), int(y_center - height / 2)

        # Get unique color for the current class
        
        color = colors.get(class_id, [0, 0, 0])
        print(color)

        # Draw rectangle and put class label
        cv2.rectangle(image, (x_min, y_min), (int(x_min + width), int(y_min + height)), color, 2)
        cv2.putText(image, class_label, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    # Save the image with bounding boxes
    cv2.imwrite(output_path, image)






def load_annotations(file_path):
    """
    Load annotations from a file in YOLO format.

    Parameters:
    - file_path: path to the annotations file.

    Returns:
    - A list of tuples containing the annotations for each bounding box.
    """
    annotations = []
    with open(file_path, 'r') as file:
        for line in file:
            # Split each line by space and convert to the appropriate types
            class_id, x_center, y_center, width, height = line.strip().split()
            class_id = int(class_id)
            x_center = float(x_center)
            y_center = float(y_center)
            width = float(width)
            height = float(height)

            # Append the annotation to the list
            annotations.append((class_id, x_center, y_center, width, height))
    
    return annotations


# image_path = 'datasets/test/images/CVC-MUSCIMA_W-01_N-10_D-ideal.png'
# annotations_path = 'datasets/test/labels/CVC-MUSCIMA_W-01_N-10_D-ideal.txt'
# output_path = 'CVC-MUSCIMA_W-01_N-10_D-ideal-ground-truth.txt'

image_path = 'MUSCIMA++/datasets_r_staff/test/images/CVC-MUSCIMA_W-01_N-10_D-symbol.png'
annotations_path = 'MUSCIMA++/datasets_r_staff/test/labels/CVC-MUSCIMA_W-01_N-10_D-symbol.txt'
# annotations_path = '/homes/iws/lq9/notation_assembly/runs/detect/predict2/labels/CVC-MUSCIMA_W-01_N-10_D-symbol.txt'
output_path = 'example_annotations/CVC-MUSCIMA_W-01_N-10_D-symbol-ground-truth.png'
annotations = load_annotations(annotations_path)  # Example: class_id, x_center, y_center, width, height
class2id = {'noteheadFull': '0', 'stem': '2', 'beam': '7', 'augmentationDot': '8', 'accidentalSharp': '9', 'accidentalFlat': '10', 'accidentalNatural': '11', 'accidentalDoubleSharp': '12', 'accidentalDoubleFlat': '13', 'restWhole': '14', 'restHalf': '15', 'restQuarter': '16', 'rest8th': '17', 'rest16th': '18', 'rest32nd': '19', 'rest64th': '20', 'multiMeasureRest': '21', 'repeat1Bar': '22', 'legerLine': '23', 'graceNoteAcciaccatura': '24', 'noteheadFullSmall': '25', 'noteheadHalfSmall': '26', 'staffSpace': '32', 'staffLine': '33', 'staff': '34', 'bracket': '35', 'brace': '36', 'staffGrouping': '37', 'barline': '38', 'barlineHeavy': '39', 'measureSeparator': '40', 'repeat': '41', 'repeatDot': '42', 'segno': '43', 'coda': '44', 'volta': '45', 'articulationStaccato': '46', 'characterDot': '47', 'articulationTenuto': '48', 'articulationAccent': '49', 'articulationMarcatoAbove': '50', 'articulationMarcatoBelow': '51', 'slur': '52', 'tie': '53', 'dynamicCrescendoHairpin': '54', 'dynamicDiminuendoHairpin': '55', 'ornament': '56', 'wiggleTrill': '57', 'ornamentTrill': '58', 'arpeggio': '59', 'glissando': '60', 'tremoloMark': '62', 'singleNoteTremolo': '63', 'multipleNoteTremolo': '64', 'tupleBracket': '65', 'tuple': '66', 'gClef': '67', 'fClef': '68', 'cClef': '69', 'keySignature': '71', 'timeSignature': '72', 'timeSigCommon': '73', 'timeSigCutCommon': '74', 'ossia': '76', 'dynamicsText': '77', 'tempoText': '78', 'instrumentName': '79', 'lyricsText': '80', 'rehearsalMark': '81', 'otherText': '82', 'characterSmallA': '83', 'characterSmallB': '84', 'characterSmallC': '85', 'characterSmallD': '86', 'characterSmallE': '87', 'characterSmallF': '88', 'characterSmallG': '89', 'characterSmallH': '90', 'characterSmallI': '91', 'characterSmallJ': '92', 'characterSmallK': '93', 'characterSmallL': '94', 'characterSmallM': '95', 'characterSmallN': '96', 'characterSmallO': '97', 'characterSmallP': '98', 'characterSmallQ': '99', 'characterSmallR': '100', 'characterSmallS': '101', 'characterSmallT': '102', 'characterSmallU': '103', 'characterSmallV': '104', 'characterSmallW': '105', 'characterSmallX': '106', 'characterSmallY': '107', 'characterSmallZ': '108', 'characterCapitalA': '109', 'characterCapitalB': '110', 'characterCapitalC': '111', 'characterCapitalD': '112', 'characterCapitalE': '113', 'characterCapitalF': '114', 'characterCapitalG': '115', 'characterCapitalH': '116', 'characterCapitalI': '117', 'characterCapitalJ': '118', 'characterCapitalK': '119', 'characterCapitalL': '120', 'characterCapitalM': '121', 'characterCapitalN': '122', 'characterCapitalO': '123', 'characterCapitalP': '124', 'characterCapitalQ': '125', 'characterCapitalR': '126', 'characterCapitalS': '127', 'characterCapitalT': '128', 'characterCapitalU': '129', 'characterCapitalV': '130', 'characterCapitalW': '131', 'characterCapitalX': '132', 'characterCapitalY': '133', 'characterCapitalZ': '134', 'numeral0': '135', 'numeral1': '136', 'numeral2': '137', 'numeral3': '138', 'numeral4': '139', 'numeral5': '140', 'numeral6': '141', 'numeral7': '142', 'numeral8': '143', 'numeral9': '144', 'barNumber': '145', 'otherNumericSign': '146', 'instrumentSpecific': '147', 'unclassified': '148', 'horizontalSpanner': '149', 'dottedHorizontalSpanner': '150', 'systemSeparator': '151', 'barlineDotted': '152', 'figuredBassText': '153', 'transpositionText': '155', 'characterOther': '156', 'breathMark': '157', 'noteheadHalf': '158', 'noteheadWhole': '159', 'flag8thUp': '160', 'flag8thDown': '161', 'flag16thUp': '162', 'flag16thDown': '163', 'flag32ndUp': '164', 'flag32ndDown': '165', 'flag64thUp': '166', 'flag64thDown': '167', 'fermataAbove': '168', 'fermataBelow': '169', 'dynamicLetterP': '170', 'dynamicLetterM': '171', 'dynamicLetterF': '172', 'dynamicLetterS': '173', 'dynamicLetterZ': '174', 'dynamicLetterR': '175', 'dynamicLetterN': '176'}
id2class = {int(value):key for key, value in class2id.items()}
draw_and_save_bounding_box(image_path, annotations, id2class, output_path)# Example usage
# get_predictions(image_path)