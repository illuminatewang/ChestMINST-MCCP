#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
纯 config 版本：从 YAML 配置文件读取所有参数，不再用命令行指定比例与路径。
本脚本把 MedMNIST 的官方 `val` 集合切成：
  1) `val_tune`（仅用于训练阶段的验证/早停/挑 best.pth） 与 `cal`（仅供 CP 校准）；
  2) 再把 `cal` 切成 `D1`（用于聚类） 与 `D2`（用于按簇估 (1-α) 分位数阈值）。

⚠️ 本脚本只“生成索引列表”（JSON + 校验用 CSV），不会移动/复制任何图片或 npz 文件。
   你在主程序里用 torch.utils.data.Subset(原 val 数据集, 索引列表) 即可“虚拟切分”。

使用方法：
    python split_val_indices_config_cn.py --config_file config.yaml

YAML 配置（示例片段，可直接加到你的 config.yaml 中）：
----------------------------------------------------
dataset: chestmnist          # MedMNIST 的数据集键
data_path: C:/path/to/data   # *.npz 所在根目录
img_size: 224                # 仅用于实例化数据集（不会影响切分）
seed: 2025                   # 随机种子（保证可复现）

val_split:                   # 本脚本读取的切分配置
  out_dir: ./splits
  val_tune_ratio: 0.2        # val -> val_tune 的比例（其余是 cal）
  d1_ratio: 0.3              # cal -> D1 的比例（其余是 D2）
----------------------------------------------------
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import yaml  # pip install pyyaml

import medmnist
from medmnist import INFO
import torchvision.transforms as transforms

def build_val_dataset(dataset: str, data_path: str, img_size: int = 224):
    if dataset not in INFO:
        raise ValueError(f"未知的数据集键：{dataset}，可选项示例：{list(INFO.keys())[:5]} ...")

    info = INFO[dataset]
    DataClass = getattr(medmnist, info["python_class"])
    eval_transform = transforms.Compose([transforms.ToTensor()])
    val_dataset = DataClass(split="val", transform=eval_transform, download=False,
                            as_rgb=True, size=img_size, root=data_path)
    return val_dataset, info

def split_indices_total(N: int, ratio: float, rng: np.random.RandomState):
    if not (0.0 < ratio < 1.0):
        raise ValueError(f"ratio 必须在 (0,1) 内，目前是 {ratio}")
    idx = np.arange(N)
    rng.shuffle(idx)
    cut = int(round(N * ratio))
    cut = max(1, min(cut, N - 1))
    A = idx[:cut].tolist(); B = idx[cut:].tolist()
    return A, B

def split_indices_from_list(idx_list: List[int], ratio: float, rng: np.random.RandomState):
    if not (0.0 < ratio < 1.0):
        raise ValueError(f"ratio 必须在 (0,1) 内，目前是 {ratio}")
    idx = np.array(idx_list, dtype=int).copy()
    rng.shuffle(idx)
    cut = int(round(len(idx) * ratio))
    cut = max(1, min(cut, len(idx) - 1))
    A = idx[:cut].tolist(); B = idx[cut:].tolist()
    return A, B

def summarize_counts(labels: np.ndarray, idx_maps: Dict[str, List[int]], out_csv: Path) -> None:
    import csv
    if labels.ndim == 1:
        labels = labels.reshape(-1, 1)
    N, C = labels.shape
    keys = list(idx_maps.keys())
    header = ["class"] + [f"{k}_pos" for k in keys] + ["total_pos"]
    rows = []
    for c in range(C):
        row = [c]
        total_pos = int(labels[:, c].sum())
        for k in keys:
            subset = labels[idx_maps[k], c]
            row.append(int(subset.sum()))
        row.append(total_pos)
        rows.append(row)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        import csv as _csv
        writer = _csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

def main():
    parser = argparse.ArgumentParser(description="读取 YAML 配置，将官方 val 切为 (val_tune, cal)，再将 cal 切为 (D1, D2)")
    parser.add_argument("--config_file", default="config.yaml", type=str, help="YAML 配置文件路径")
    args = parser.parse_args()

    with open(args.config_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    dataset = cfg.get("dataset", "chestmnist")
    data_path = cfg.get("data_path", None)
    if not data_path:
        raise ValueError("config.yaml 缺少 data_path 字段（指向 *.npz 根目录）。")
    img_size = int(cfg.get("img_size", 224))
    seed = int(cfg.get("seed", 2025))

    val_split = cfg.get("val_split", {})
    out_dir = val_split.get("out_dir", "./splits")
    val_tune_ratio = float(val_split.get("val_tune_ratio", 0.20))
    d1_ratio = float(val_split.get("d1_ratio", 0.30))

    rng = np.random.RandomState(seed)
    val_dataset, info = build_val_dataset(dataset, data_path, img_size)
    labels = np.array(val_dataset.labels, dtype=np.int64)
    N_val = len(val_dataset)
    if N_val < 10:
        raise RuntimeError(f"val 样本过少（N_val={N_val}），请检查 data_path 是否正确：{data_path}")

    val_tune_idx, cal_idx = split_indices_total(N_val, val_tune_ratio, rng)
    D1_idx, D2_idx = split_indices_from_list(cal_idx, d1_ratio, rng)

    S_val_tune = set(map(int, val_tune_idx)); S_cal = set(map(int, cal_idx))
    S_D1 = set(map(int, D1_idx)); S_D2 = set(map(int, D2_idx))
    assert S_val_tune.isdisjoint(S_cal), "val_tune 与 cal 不应重叠！"
    assert S_D1.isdisjoint(S_D2), "D1 与 D2 不应重叠！"
    assert S_D1.union(S_D2) == S_cal, "D1 ∪ D2 应等于 cal！"
    all_idx = S_val_tune.union(S_cal)
    assert min(all_idx) >= 0 and max(all_idx) < N_val, "索引越界，请检查切分逻辑！"

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{dataset}_val_splits_seed{seed}.json"
    csv_path = out_dir / f"{dataset}_val_splits_seed{seed}_counts.csv"

    payload = {
        "val_tune": val_tune_idx,
        "cal": cal_idx,
        "D1": D1_idx,
        "D2": D2_idx,
        "meta": {
            "dataset": dataset,
            "seed": seed,
            "val_tune_ratio": val_tune_ratio,
            "d1_ratio": d1_ratio,
            "N_val": N_val,
            "N_val_tune": len(val_tune_idx),
            "N_cal": len(cal_idx),
            "N_D1": len(D1_idx),
            "N_D2": len(D2_idx),
            "python_class": info.get("python_class", None),
            "label": info.get("label", None),
        }
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    idx_maps = {"val_tune": val_tune_idx, "cal": cal_idx, "D1": D1_idx, "D2": D2_idx}
    summarize_counts(labels, idx_maps, csv_path)

    print("[OK] 已保存索引到：", json_path)
    print("[OK] 已保存各 split 的逐类阳性计数到：", csv_path)
    print("Meta:", payload["meta"])
    print("提示：训练阶段只用 val_tune；CP 在 cal 里再用 D1/D2，test 仅做最终评估。")

if __name__ == "__main__":
    main()
