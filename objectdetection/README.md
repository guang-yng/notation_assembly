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
```bash
python train.py
```

Set the yaml file in `train.py` for different datasets.
- `data.yaml`: dataset with staff.
- `data_staff_removed.yaml`: dataset without staff.
- `data_staff_removed_20.yaml`: dataset without staff, containing 20 classes only.
- `data_staff_removed_20_crop.yaml`: dataset without staff, containing 20 classes only, cropped.

To run on a new dataset, copy the yaml file and modify for your convenience.