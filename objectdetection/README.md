# Music Object Detection

## Data Generation

Use the scrip `data_gen.py` to generate new dataset.
```
usage: data_gen.py [-h] [-d DATA] [--classes CLASSES] [--save_dir SAVE_DIR] [--seed SEED] [--crop_times CROP_TIMES]

optional arguments:
  -h, --help            show this help message and exit
  -d DATA, --data DATA  data directory
  --classes CLASSES     the path to the musima classes definition xml file
  --save_dir SAVE_DIR   The output directory
  --seed SEED           random seed
  --crop_times CROP_TIMES
                        number of crops for each image
```
Check the script for details.
Notice that if you want to use `classes` option, you need to modify line 210-218.
Default setting is to use the 20 classes mentioned in the paper.

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