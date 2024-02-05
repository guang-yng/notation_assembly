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


## Model Training
'''bash
python -m effnet.train --model_config configs/effnet/MLP32.yaml
'''

To test the model, pass *--test_only* to the above command.
