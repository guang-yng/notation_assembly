import torch
from munglinker.data_pool import load_munglinker_data

if __name__ == "__main__":
    mung_root = "data/MUSCIMA++/v2.0/data/"
    image_root = "data/MUSCIMA++/v2.0/data/images/"
    split_file = "splits/mob_split.yaml"
    config_file = "configs/muscima_bboxes.yaml"
    data = load_munglinker_data(
        mung_root=mung_root,
        images_root=image_root,
        split_file=split_file,
        config_file=config_file,
        load_training_data=True,
        load_validation_data=True,
        load_test_data=False
    )
    assert data['train'][0]['targets'][0] == 1.0
    assert data['valid'][33]['mungos_to'][0].bounding_box == (286, 223, 405, 3349)
    print("DONE")