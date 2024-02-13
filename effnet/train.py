import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import f1_score, precision_recall_fscore_support, PrecisionRecallDisplay, precision_recall_curve
import matplotlib.pyplot as plt

from munglinker.data_pool import load_munglinker_data

from utils.constants import node_classes_dict
from configs.effnet.default import get_cfg_defaults
from effnet.net import MLP

import argparse
import os
import tqdm


def train(args, data, cfg, device, model):
    with open(f"{args.output_dir}/{args.exp_name}/config.yaml", 'w') as f:
        f.write(cfg.dump())
        
    model.train()
    if cfg.TRAIN.OPTIMIZER == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.LEARNING_RATE)
    else:
        raise ValueError(f"Optimizer {cfg.TRAIN.OPTIMIZER} is not supported")
    
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(cfg.TRAIN.POS_WEIGHT))
    loader = DataLoader(data['train'], batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=cfg.SYSTEM.NUM_WORKERS)

    for epoch in range(args.load_epochs, cfg.TRAIN.NUM_EPOCHS):
        corr = 0
        total = 0
        for batch in tqdm.tqdm(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch['label'])
            loss.backward()
            optimizer.step()

            pred = torch.sigmoid(output) > 0.5
            corr += (pred == batch['label']).sum().item()
            total += len(batch['label'])
        print(f"Epoch {epoch+1} accuracy: {corr/total}")

        if (epoch+1) % cfg.TRAIN.EVAL_FREQUENCY == 0:
            with torch.no_grad():
                acc, F1 = eval(args, data['valid'], cfg, device, model)
            print(f"Epoch {epoch+1} validation accuracy: {acc}, F1: {F1}")
        if (epoch+1) % cfg.TRAIN.SAVE_FREQUENCY == 0 and (epoch+1) != cfg.TRAIN.NUM_EPOCHS:
            torch.save(model.state_dict(), f"{args.output_dir}/{args.exp_name}/model_ep{epoch+1}.pth")

    torch.save(model.state_dict(), f"{args.output_dir}/{args.exp_name}/model_final.pth")

    return model


def eval(args, data, cfg, device, model, plot_PRC=False):
    model.eval()
    loader = DataLoader(data, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False, num_workers=cfg.SYSTEM.NUM_WORKERS)

    corr = 0
    total = 0
    preds = []
    labels = []
    logits = []
    for batch in tqdm.tqdm(loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        output = model(batch)

        logit = torch.sigmoid(output)
        logits += logit.squeeze().tolist()
        pred = logit > 0.5
        preds += pred.squeeze().tolist()
        labels += batch['label'].squeeze().tolist()
        corr += (pred == batch['label']).sum().item()
        total += len(batch['label'])

    precision, recall, F1, _ = precision_recall_fscore_support(labels, preds)
    if plot_PRC:
        prec, rec, _ = precision_recall_curve(labels, logits)
        plt.plot(rec, prec)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.show()
        plt.savefig(f"{args.output_dir}/{args.exp_name}/PRC.png")
    print(f"Precision: {precision}, Recall: {recall}")
    model.train()
    return corr/total, F1


def main(args, data, cfg, device):
    if cfg.MODEL.MODE == "MLP":
        model = MLP(cfg)
    else:
        raise ValueError(f"Model {cfg.TRAIN.MODEL} is not supported")
    model.to(device)
    
    if args.load_epochs > 0:
        model.load_state_dict(torch.load(f"{args.output_dir}/{args.exp_name}/model_ep{args.load_epochs}.pth"))
    elif args.load_epochs == 0 or args.test_only:
        # Load the final model if it exists, otherwise load the last model
        if os.path.exists(f"{args.output_dir}/{args.exp_name}/model_final.pth"):
            model.load_state_dict(torch.load(f"{args.output_dir}/{args.exp_name}/model_final.pth"))
        else:
            model_files = [f for f in os.listdir(f"{args.output_dir}/{args.exp_name}") if f.startswith("model_ep") and f.endswith(".pth")]
            if len(model_files) > 0:
                model_files = sorted(model_files, key=lambda x: int(x.split("_")[-1].split(".")[0][2:]), reverse=True)
                model.load_state_dict(torch.load(f"{args.output_dir}/{args.exp_name}/{model_files[0]}"))    
    if args.test_only:
        acc, F1 = eval(args, data['test'], cfg, device, model, plot_PRC=True)
        print(f"Accuracy: {acc}, F1: {F1}")
    else:
        model = train(args, data, cfg, device, model)
        acc, F1 = eval(args, data['valid'], cfg, device, model)
        print(f"Final accuracy: {acc}, F1: {F1}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mung_root', default="data/MUSCIMA++/v2.0/data/", help="The root directory of the MUSCIMA++ dataset")
    parser.add_argument('-i', '--image_root', default="data/MUSCIMA++/v2.0/data/images/", help="The root directory of the MUSCIMA++ images")
    parser.add_argument('-s', '--split_file', default="splits/mob_split.yaml", help="The split file")
    parser.add_argument('--data_config', default="configs/muscima_bboxes.yaml", help="The config file")
    parser.add_argument('--model_config', type=str)
    parser.add_argument('--output_dir', type=str, default="outputs")
    parser.add_argument('--exp_name', type=str)
    parser.add_argument('--test_only', action="store_true")
    parser.add_argument('--load_epochs', type=int, default=-1)
    parser.add_argument('--opts', default=[], nargs=argparse.REMAINDER, help="options to overwrite the config")

    args = parser.parse_args()

    cfg = get_cfg_defaults()
    if args.test_only:
        cfg.merge_from_file(os.path.join(args.output_dir, args.exp_name, "config.yaml"))
    if args.model_config and not args.test_only:
        cfg.merge_from_file(args.model_config)
    cfg.merge_from_list(args.opts)

    if not args.test_only:
        data = load_munglinker_data(
            mung_root=args.mung_root,
            images_root=args.image_root,
            split_file=args.split_file,
            config_file=cfg.DATA.DATA_CONFIG,
            load_training_data=True,
            load_validation_data=True,
            load_test_data=False,
        )
    else:
        data = load_munglinker_data(
            mung_root=args.mung_root,
            images_root=args.image_root,
            split_file=args.split_file,
            config_file=cfg.DATA.DATA_CONFIG,
            load_training_data=False,
            load_validation_data=False,
            load_test_data=True,
        )

    if cfg.SYSTEM.NUM_GPUS > 0:
        if not torch.cuda.is_available():
            raise ValueError("No GPU available")
        device = "cuda"
    else:
        device = "cpu"
    
    import random
    random.seed(cfg.SYSTEM.SEED)
    import numpy as np
    np.random.seed(cfg.SYSTEM.SEED)
    torch.manual_seed(cfg.SYSTEM.SEED)

    os.makedirs(f"{args.output_dir}/{args.exp_name}", exist_ok=True)

    main(args, data, cfg, device)


