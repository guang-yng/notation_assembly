# notation_assembly

## Data Preparation
The MUSCIMA++ dataset is available at [here](https://ufal.mff.cuni.cz/muscima).

To set up the environment, please run:
```bash
pip install -r requirements.txt
```


Generate datasets using the following commands:
```bash
python utils/data_gen.py
```

Check codes for possible options.


## Notation Assembler Training
```bash
python -m effnet.train --model_config configs/effnet/MLP32_nogrammar.yaml --exp_name MLP32_nogrammar.yaml
```

To test the model, pass *--test_only* or *--val_only* to the above command.

## Object Detection Data & Training
Check [README](objectdetection/README.md) for details.


## Other scripts

### Generate sample images for each class

```bash
python utils/visualize_classes.py
```
Run `python utils/visualize_classes.py -h` to get available arguments.
