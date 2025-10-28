#打印图像
from pathlib import Path
import json
import csv
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize


import numpy as np
import torchvision.transforms as transforms
def painting(config: dict,ACC,AUC,y_true,y_pred):
    # ======== 保存评估结果与 ROC 可视化 ========
    # 统一输出目录 & 命名（沿用 train.py 的模式）
    Path(config['output_path']).mkdir(parents=True, exist_ok=True)
    save_prefix = f"{config['output_path']}/{config['dataset']}_{config['img_size']}_{config['training_procedure']}_{config['architecture']}_s{config['seed']}"

    # --- 写入 metrics 到 JSON & CSV ---
    metrics = {
        "dataset": config['dataset'],
        "img_size": config['img_size'],
        "training_procedure": config['training_procedure'],
        "architecture": config['architecture'],
        "seed": int(config['seed']),
        "ACC": float(ACC),
        "AUC": float(AUC) if isinstance(AUC, (int, float)) else AUC,
    }
    # JSON
    with open(f"{save_prefix}_metrics.json", "w", encoding="utf-8") as jf:
        json.dump(metrics, jf, ensure_ascii=False, indent=2)

    # CSV（追加或新建）
    csv_path = f"{save_prefix}_metrics.csv"
    header = ["dataset", "img_size", "training_procedure", "architecture", "seed", "ACC", "AUC"]
    row = [config['dataset'], config['img_size'], config['training_procedure'], config['architecture'], config['seed'], ACC,
           AUC]
    try:
        need_header = not Path(csv_path).exists()
        with open(csv_path, "a", newline="", encoding="utf-8") as cf:
            writer = csv.writer(cf)
            if need_header:
                writer.writerow(header)
            writer.writerow(row)
    except PermissionError:
        print("\t[WARN] metrics.csv 正在被占用，跳过 CSV 写入。")

    # --- 绘制并保存 ROC 曲线 ---
    # 仅在非 kNN 且任务可计算 ROC 时绘图
    if config['training_procedure'] != 'kNN':
        y_true_np = y_true.cpu().numpy()
        y_pred_np = y_pred.cpu().numpy()

        try:
            if config['task'] == "multi-label, binary-class":
                # 多标签：为每个类别画一张 ROC
                num_classes = y_true_np.shape[1]
                for c in range(num_classes):
                    fpr, tpr, _ = roc_curve(y_true_np[:, c], y_pred_np[:, c])
                    roc_auc = auc(fpr, tpr)

                    plt.figure()
                    plt.plot(fpr, tpr, lw=2, label=f"Class {c} (AUC={roc_auc:.4f})")
                    plt.plot([0, 1], [0, 1], lw=1, linestyle="--")
                    plt.xlabel("False Positive Rate")
                    plt.ylabel("True Positive Rate")
                    plt.title(f"ROC Curve - Class {c}")
                    plt.legend(loc="lower right")
                    plt.tight_layout()
                    plt.savefig(f"{save_prefix}_roc_class{c}.png", dpi=200)
                    plt.close()

            else:
                # 单标签：如果是二分类，直接画；如果是多类，做 one-vs-rest
                n_unique = len(np.unique(y_true_np.astype(int)))
                if n_unique == 2:
                    # 二分类
                    fpr, tpr, _ = roc_curve(y_true_np.ravel(),
                                            y_pred_np[:, 1].ravel() if y_pred_np.shape[1] > 1 else y_pred_np.ravel())
                    roc_auc = auc(fpr, tpr)

                    plt.figure()
                    plt.plot(fpr, tpr, lw=2, label=f"AUC={roc_auc:.4f}")
                    plt.plot([0, 1], [0, 1], lw=1, linestyle="--")
                    plt.xlabel("False Positive Rate")
                    plt.ylabel("True Positive Rate")
                    plt.title("ROC Curve")
                    plt.legend(loc="lower right")
                    plt.tight_layout()
                    plt.savefig(f"{save_prefix}_roc.png", dpi=200)
                    plt.close()

                else:
                    # 多类：one-vs-rest
                    classes = np.unique(y_true_np.astype(int)).tolist()
                    y_true_bin = label_binarize(y_true_np.astype(int), classes=classes)

                    plt.figure()
                    for idx, cls in enumerate(classes):
                        fpr, tpr, _ = roc_curve(y_true_bin[:, idx], y_pred_np[:, idx])
                        roc_auc = auc(fpr, tpr)
                        plt.plot(fpr, tpr, lw=1.5, label=f"Class {cls} (AUC={roc_auc:.4f})")

                    plt.plot([0, 1], [0, 1], lw=1, linestyle="--")
                    plt.xlabel("False Positive Rate")
                    plt.ylabel("True Positive Rate")
                    plt.title("ROC Curve (One-vs-Rest)")
                    plt.legend(loc="lower right", fontsize=8)
                    plt.tight_layout()
                    plt.savefig(f"{save_prefix}_roc_multiclass.png", dpi=200)
                    plt.close()

        except Exception as e:
            print(f"\t[WARN] 生成 ROC 曲线失败：{e}")
    # ======== 保存评估结果与 ROC 可视化 END ========

