# Music Object Detection

## Data Generation

Use the scrip `data_gen.py` to generate new dataset.
```
usage: data_gen.py [-h] [-d DATA] [--image_dir IMAGE_DIR] [--classes CLASSES] [--save_dir SAVE_DIR] [--save_config SAVE_CONFIG] [--seed SEED] [--crop_times CROP_TIMES]                   

optional arguments:
  -h, --help            show this help message and exit
  -d DATA, --data DATA  data directory of annotations
  --image_dir IMAGE_DIR
                        data directory of images
  --classes CLASSES     The path to the musima classes definition xml file. If set to '20', 20 restricted classes are used. If set to 'essential', essential classes are used.
  --save_dir SAVE_DIR   The output directory
  --save_config SAVE_CONFIG
                        The path to save yaml file
  --seed SEED           random seed
  --crop_times CROP_TIMES
                        number of crops for each image
```
Check the script for default values.

Here are some examples below:

Generate the cropped dataset with staff removed:
```bash
python data_gen.py --save_dir MUSCIMA++/datasets_r_staff_crop --save_config data_staff_removed_crop.yaml
```

Generate the uncropped dataset with staff removed and 20 classes only:
```bash
python data_gen.py --classes 20 --save_dir MUSCIMA++/datasets_r_staff_20 --save_config data_staff_removed_20.yaml --crop_times 0
```

Generate the uncropped dataset with staff:
```bash
python data_gen.py --image_dir MUSCIMA++/datasets_w_staff/images --save_dir MUSCIMA++/datasets_w_staff --save_config data_staff.yaml --crop_times 0
```

Generating cropped dataset with staff removed and essential classes:
```bash
python data_gen.py --classes essential --save_dir MUSCIMA++/datasets_r_staff_essential_crop --save_config data_staff_removed_essesntial_crop.yaml
```

## Training

To run the pipeline:
Run `train.py` to train a model:
```
usage: train.py [-h] [--config CONFIG] [-e EPOCHS] [-b BATCH_SIZE] [-m MODEL] [-i IMGSZ] [--output_dir OUTPUT_DIR]

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG       The yaml config file for dataset.
  -e EPOCHS, --epochs EPOCHS
                        Number of training epochs.
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        Batch size.
  -m MODEL, --model MODEL
                        The model weights to use.
  -i IMGSZ, --imgsz IMGSZ
                        The image size of YOLO. Choose from [640, 1280, 1920].
  --output_dir OUTPUT_DIR
                        The directory to save training info & models.
```

Here are some default config files:
- `data.yaml`: dataset with staff.
- `data_staff_removed.yaml`: dataset without staff.
- `data_staff_removed_20.yaml`: dataset without staff, containing 20 classes only.
- `data_staff_removed_20_crop.yaml`: dataset without staff, containing 20 classes only, cropped.
- `data_staff_removed_crop.yaml`: dataset without staff, cropped.
- `data_staff_removed_essential_crop.yaml`: dataset without staff, containing essential classes, cropped.

## Inference

To get the predictions on test dataset, run the following script:
```
usage: infer.py [-h] [--model MODEL] [--data DATA] [--classes CLASSES] [--visualize] [--grids] [--batch_size BATCH_SIZE]
                [--save_dir SAVE_DIR]

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         The model to load.
  --data DATA           The dataset path.
  --classes CLASSES     The path to the musima classes definition xml file. If set to '20', 20 restricted classes are used. If set to
                        'essential', essential classes are used.
  --visualize           Whether visualize the result
  --grids               Whether to visualize the girds. Only valid when --visualize is set.
  --batch_size BATCH_SIZE
                        The batch size for inference.
  --save_dir SAVE_DIR   The directory to save results
```

For example, to run inference on essential classes and visualize the result:
```bash
python infer.py --classes essential --visualize
```