"""
xAILab
Chair of Explainable Machine Learning
Otto-Friedrich University of Bamberg

@description:
Main script to train and evaluate a model on the specified dataset of the MedMNIST+ collection.
"""
# Import packages
import argparse
import yaml
import torch
import timm
import time
import medmnist
import random
import numpy as np
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from medmnist import INFO

# Import custom modules
from train import train
from evaluate import evaluate
from utils import calculate_passed_time, seed_worker

##聚类CP########################################################################################
import os, numpy as np
from cp import fit_clustered_cp_return_classwise_tau  # 你 cp.py 里的函数
import timm, torch.nn as nn
from torchvision.models import alexnet, AlexNet_Weights

def run_cp_calibration(config, D1_loader, D2_loader):
    """
    载入 best.pth，基于 D1/D2 校准聚类CP，保存 calibrator npz，并把路径写进 config。
    """
    # 1) 重建模型并加载 best 权重（与 evaluate.py 同步）
    if config['architecture'] == 'alexnet':
        model = alexnet(weights=AlexNet_Weights.DEFAULT)
        model.classifier[6] = nn.Linear(4096, config['num_classes'])
    else:
        model = timm.create_model(config['architecture'], pretrained=True, num_classes=config['num_classes'])

    checkpoint_file = f"{config['output_path']}/{config['dataset']}_{config['img_size']}_{config['training_procedure']}_{config['architecture']}_s{config['seed']}_best.pth"
    checkpoint = torch.load(checkpoint_file, map_location='cpu')
    model.load_state_dict(checkpoint)  # 同 evaluate.py 的加载方式 :contentReference[oaicite:3]{index=3}

    model = model.to(config['device'])
    model.eval(); model.requires_grad_(False)

    # 2) 运行聚类CP（手动 M）
    cal = fit_clustered_cp_return_classwise_tau(
        model, config['device'], D1_loader, D2_loader,
        alpha=float(config.get('cp_alpha', 0.10)),
        n_clusters=int(config.get('cluster_K', 3)),
        random_state=config['seed'],
    )

    # 3) 保存 calibrator —— 改为 JSON（可选再导出一个 classes.csv）
    os.makedirs(config['output_path'], exist_ok=True)
    base = f"{config['dataset']}_{config['img_size']}_{config['training_procedure']}_{config['architecture']}_s{config['seed']}_cp_calibrator"
    cal_file = os.path.join(config['output_path'], base + ".json")

    # 组装成纯 Python 基本类型，便于 JSON 存储
    cal_json = {
        "tau_per_class": cal["tau_per_class"].tolist(),
        "cluster_map": cal["cluster_map"].tolist(),
        "q_per_cluster": cal["q_per_cluster"].tolist(),
        "q_global": float(cal["q_global"]),
        "alpha": float(cal["alpha"]),
    }

    import json
    with open(cal_file, "w", encoding="utf-8") as f:
        json.dump(cal_json, f, ensure_ascii=False, indent=2)

    # （可选）导出一份人类可读的 CSV：每类的 τ 和簇号
    try:
        from medmnist import INFO
        label_dict = INFO[config['dataset']]['label']
        idxs = sorted(map(int, label_dict.keys()))
        class_names = [label_dict[str(i)] for i in idxs]
    except Exception:
        class_names = [f"c{i}" for i in range(len(cal["tau_per_class"]))]

    classes_csv = os.path.join(config['output_path'], base + "_classes.csv")
    import csv
    with open(classes_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class_idx", "class_name", "cluster_id", "tau"])
        for j, (name, cid, tauj) in enumerate(zip(class_names, cal["cluster_map"], cal["tau_per_class"])):
            w.writerow([j, name, int(cid), float(tauj)])

    config['cp_calibrator_file'] = cal_file
    print(f"[CP] Saved calibrator to: {cal_file}")
    print(f"[CP] Also saved class-wise CSV to: {classes_csv}")

################################################################################################################

def main(config: dict):
    """
    Main function to train and evaluate a model on the specified dataset.

    :param config: Dictionary containing the parameters and hyperparameters.
    """

    # Start code
    start_time = time.time()
    print("\tRun Details:")
    print("\t\tDataset: {}".format(config['dataset']))
    print("\t\tImage size: {}".format(config['img_size']))
    print("\t\tTraining procedure: {}".format(config['training_procedure']))
    print("\t\tArchitecture: {}".format(config['architecture']))
    print("\t\tSeed: {}".format(config['seed']))

    # Seed the training and data loading so both become deterministic
    print("\tSeed:")
    if config['architecture'] == 'alexnet':
        torch.backends.cudnn.benchmark = True  # Enable the benchmark mode in cudnn
        torch.backends.cudnn.deterministic = False  # Disable cudnn to be deterministic
        torch.use_deterministic_algorithms(False)  # Disable only deterministic algorithms

    else:
        torch.backends.cudnn.benchmark = False  # Disable the benchmark mode in cudnn
        torch.backends.cudnn.deterministic = True  # Enable cudnn to be deterministic

        if config['architecture'] == 'samvit_base_patch16':
            torch.use_deterministic_algorithms(False)  #True, warn_only=True

        else:
            torch.use_deterministic_algorithms(False)  # Enable only deterministic algorithms #True

    torch.manual_seed(config['seed'])  # Seed the pytorch RNG for all devices (both CPU and CUDA)
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    g = torch.Generator()
    g.manual_seed(config['seed'])

    # Extract the dataset and its metadata
    info = INFO[config['dataset']]
    config['task'], config['in_channel'], config['num_classes'] = info['task'], info['n_channels'], len(info['label'])
    DataClass = getattr(medmnist, info['python_class'])

    # Create the data transforms and normalize with imagenet statistics
    if config['architecture'] == 'alexnet':
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)  # Use ImageNet statistics
    else:
        m = timm.create_model(config['architecture'], pretrained=True)
        mean, std = m.default_cfg['mean'], m.default_cfg['std']

    total_padding = max(0, 224 - config['img_size'])
    padding_left, padding_top = total_padding // 2, total_padding // 2
    padding_right, padding_bottom = total_padding - padding_left, total_padding - padding_top

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
        transforms.Pad((padding_left, padding_top, padding_right, padding_bottom), fill=0, padding_mode='constant')  # Pad the image to 224x224
    ])

    # Create the datasets
    train_dataset = DataClass(split='train', transform=data_transform, download=False, as_rgb=True,
                              size=config['img_size'],
                              root=config['data_path'])
    val_dataset = DataClass(split='val', transform=data_transform, download=False, as_rgb=True, size=config['img_size'],
                            root=config['data_path'])
    test_dataset = DataClass(split='test', transform=data_transform, download=False, as_rgb=True,
                             size=config['img_size'],
                             root=config['data_path'])
    # Create the dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0,
                              worker_init_fn=seed_worker, generator=g)
    train_loader_at_eval = DataLoader(train_dataset, batch_size=config['batch_size_eval'], shuffle=False, num_workers=4,
                                      worker_init_fn=seed_worker, generator=g)  # knn使用
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size_eval'], shuffle=False, num_workers=4,
                            worker_init_fn=seed_worker, generator=g)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size_eval'], shuffle=False, num_workers=4,
                             worker_init_fn=seed_worker, generator=g)
## val_loader分割###################################################################################
    import json
    from torch.utils.data import Subset

    splits_cfg = config.get('val_split', {})
    if splits_cfg.get('use_splits', False):
        with open(splits_cfg['split_file'], 'r', encoding='utf-8') as f:
            splits = json.load(f)

        idx_val_tune = splits['val_tune']  # list[int]
        idx_D1 = splits['D1']  # list[int]
        idx_D2 = splits['D2']  # list[int]

        val_tune_dataset = Subset(val_dataset, idx_val_tune)
        D1_dataset = Subset(val_dataset, idx_D1)
        D2_dataset = Subset(val_dataset, idx_D2)

        val_tune_loader = DataLoader(val_tune_dataset, batch_size=config['batch_size_eval'],
                                     shuffle=False, num_workers=4, worker_init_fn=seed_worker, generator=g)
        D1_loader = DataLoader(D1_dataset, batch_size=config['batch_size_eval'],
                               shuffle=False, num_workers=4, worker_init_fn=seed_worker, generator=g)
        D2_loader = DataLoader(D2_dataset, batch_size=config['batch_size_eval'],
                               shuffle=False, num_workers=4, worker_init_fn=seed_worker, generator=g)
    else:
        # 回退：无 splits 就沿用原来的 val_loader 当验证集
        val_tune_loader = val_loader
###########################################################################################################
    # # Run the training
    # if config['training_procedure'] == 'endToEnd' or config['training_procedure'] == 'linearProbing':
    #     train(config, train_loader, val_tune_loader)
    # elif config['training_procedure'] == 'kNN':
    #     pass
    # else:
    #     raise ValueError("The specified training procedure is not supported.")

    # Run the Conditional Conformal Prediction
    if splits_cfg.get('use_splits', False):
        # 只有 endToEnd/linearProbing（有概率输出）才做 CP
        if config['training_procedure'] in ('endToEnd', 'linearProbing'):
            run_cp_calibration(config, D1_loader, D2_loader)  # 这里会把路径写到 config 里
        else:
            print("[CP] Skip calibration for kNN (no probabilistic scores).")

    # Run the evaluation
    evaluate(config, train_loader_at_eval, test_loader)

    print(f"\tFinished current run.")
    # Stop the time
    end_time = time.time()
    hours, minutes, seconds = calculate_passed_time(start_time, end_time)
    print("\tElapsed time: {:0>2}:{:0>2}:{:05.2f}".format(hours, minutes, seconds))


if __name__ == '__main__':
    # Read out the command line parameters.
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", default="config.yaml", type=str, help="Path to the configuration file to use.")
    parser.add_argument("--dataset", required=False, type=str, help="Which dataset to use.")
    parser.add_argument("--img_size", required=False, type=int, help="Which image size to use.")
    parser.add_argument("--training_procedure", required=False, type=str, help="Which training procedure to use.")
    parser.add_argument("--architecture", required=False, type=str, help="Which architecture to use.")
    parser.add_argument("--seed", required=False, type=int, help="Which seed was used during training.")

    args = parser.parse_args()
    config_file = args.config_file

    # Load the parameters and hyperparameters of the configuration file
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Adapt to the command line arguments
    if args.dataset:
        config['dataset'] = args.dataset

    if args.img_size:
        config['img_size'] = args.img_size

    if args.training_procedure:
        config['training_procedure'] = args.training_procedure

    if args.architecture:
        config['architecture'] = args.architecture

    # If a seed is specified, overwrite the seed in the config file
    if args.seed:
        config['seed'] = args.seed

    # Run
    main(config)
