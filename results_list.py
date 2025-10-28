
# ======== 保存每张图片的预测概率 & 真实标签（可选附带二值/集合预测） ========
from pathlib import Path
import os, csv, json
import numpy as np
from medmnist import INFO

def _load_cp_calibrator(cp_file: str):
    """
    支持 .npz 或 .json 的 CP 校准器加载。
    返回：tau(np.ndarray[C]), alpha(float or None),
         cluster_map(np.ndarray[C] or None),
         q_per_cluster(np.ndarray[M] or None),
         q_global(float or None)
    """
    ext = os.path.splitext(cp_file)[1].lower()
    if ext == ".npz":
        cal = np.load(cp_file)
        tau = cal["tau_per_class"]
        alpha = float(cal["alpha"][0]) if "alpha" in cal.files else None
        cluster_map = cal["cluster_map"] if "cluster_map" in cal.files else None
        q_per_cluster = cal["q_per_cluster"] if "q_per_cluster" in cal.files else None
        q_global = float(cal["q_global"][0]) if "q_global" in cal.files else None
        return tau, alpha, cluster_map, q_per_cluster, q_global

    elif ext == ".json":
        with open(cp_file, "r", encoding="utf-8") as f:
            d = json.load(f)
        tau = np.asarray(d["tau_per_class"], dtype=float)
        alpha = float(d["alpha"]) if "alpha" in d else None
        cluster_map = np.asarray(d.get("cluster_map", []), dtype=int) if "cluster_map" in d else None
        q_per_cluster = np.asarray(d.get("q_per_cluster", []), dtype=float) if "q_per_cluster" in d else None
        q_global = float(d["q_global"]) if "q_global" in d else None
        return tau, alpha, cluster_map, q_per_cluster, q_global

    else:
        raise ValueError(f"Unsupported calibrator file: {cp_file}")

def _save_per_class_cp_coverage(class_names, y_true_np, pred_bin_cp, save_prefix):
    """
    直接基于数组计算并落盘每类的正样本覆盖率（不依赖 pandas）：
      coverage_y = (# y类真实为1且被CP命中的样本) / (# y类真实为1的样本)
    会生成：{save_prefix}_per_class_cp_coverage.csv
    """
    out_path = f"{save_prefix}_per_class_cp_coverage.csv"
    total_pos = 0
    total_cov = 0

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        import csv
        writer = csv.writer(f)
        writer.writerow(["class", "pos_count", "covered_pos_count", "coverage"])

        C = len(class_names)
        for j in range(C):
            tcol = (y_true_np[:, j] == 1)
            pcol = (pred_bin_cp[:, j] == 1)
            denom = int(tcol.sum())
            num = int((tcol & pcol).sum())
            cov = (num / denom) if denom > 0 else float("nan")

            total_pos += denom
            total_cov += num

            writer.writerow([class_names[j], denom, num, cov])

    overall = (total_cov / total_pos) if total_pos > 0 else float("nan")
    if overall == overall:
        print(f"\t[CP] per-class coverage saved: {out_path} | overall={overall:.6f}")
    else:
        print(f"\t[CP] per-class coverage saved: {out_path} | overall=NaN")

def _save_per_class_threshold_coverage(class_names, y_true_np, y_pred_np, thr, save_prefix):
    """
    基线（无CP）每类正样本覆盖率，使用固定阈值 thr：
      coverage_y = (# y类真实为1且 pred>=thr 的样本) / (# y类真实为1的样本)
    会生成：{save_prefix}_per_class_thr{thr}_coverage.csv，并在控制台打印 overall@thr 以及每类覆盖率。
    """
    pred_bin = (y_pred_np >= float(thr)).astype(int)  # [N, C]
    thr_str = f"{thr:.2f}"
    out_path = f"{save_prefix}_per_class_thr{thr_str}_coverage.csv"

    total_pos = 0
    total_cov = 0
    per_class_lines = []

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "pos_count", "covered_pos_count", "coverage"])

        C = len(class_names)
        for j in range(C):
            tcol = (y_true_np[:, j] == 1)
            pcol = (pred_bin[:, j] == 1)
            denom = int(tcol.sum())
            num = int((tcol & pcol).sum())
            cov = (num / denom) if denom > 0 else float("nan")

            total_pos += denom
            total_cov += num

            writer.writerow([class_names[j], denom, num, cov])
            if denom > 0 and cov == cov:
                per_class_lines.append(f"{class_names[j]}:{cov:.3f}")
            elif denom == 0:
                per_class_lines.append(f"{class_names[j]}:NaN(无正样本)")
            else:
                per_class_lines.append(f"{class_names[j]}:NaN")

    overall = (total_cov / total_pos) if total_pos > 0 else float("nan")
    if overall == overall:
        print(f"\t[BASE] per-class coverage saved: {out_path} | overall@{thr_str}={overall:.6f}")
    else:
        print(f"\t[BASE] per-class coverage saved: {out_path} | overall@{thr_str}=NaN")
    # 打印每类覆盖率（简洁一行）
    print("\t[BASE] per-class coverage@", thr_str, "=>", " | ".join(per_class_lines))

def save_results(config: dict, y_true, y_pred):
    """
    保存逐样本预测结果到两份CSV/一份JSONL：
      1) *_per_sample_predictions.csv           （固定阈值基线）
         表头统一为：[index, exact_match@thr, contains_all_true@thr, num_pos_true, num_pos_pred@thr, true:*, prob:*, pred@thr:*]
      2) *_per_sample_predictions_withCP.csv    （若存在CP校准器）
         表头统一为：[index, exact_match@CP, contains_all_true@CP, num_pos_true, num_pos_pred@CP, true:*, prob:*, pred@CP:*]
      3) *_per_sample_predictions.jsonl         （true/prob 便于脚本处理）
    """
    # 1) 转成 numpy
    y_true_np = y_true.detach().cpu().numpy()
    y_pred_np = y_pred.detach().cpu().numpy()

    # 2) 类名（用于更易读的表头）
    label_dict = INFO[config['dataset']]['label']  # 可能是 {"0":"...", "1":"..."} 这种
    try:
        idxs = sorted(map(int, label_dict.keys()))
        class_names = [label_dict[str(i)] for i in idxs]
    except Exception:
        # 兼容 key 已是 int 的情况
        class_names = [label_dict[k] for k in sorted(label_dict.keys())]
    num_classes = len(class_names)

    # 3) 输出路径与文件名前缀
    Path(config['output_path']).mkdir(parents=True, exist_ok=True)
    save_prefix = f"{config['output_path']}/{config['dataset']}_{config['img_size']}_{config['training_procedure']}_{config['architecture']}_s{config['seed']}"

    # 4) 写 CSV（逐样本：固定阈值基线）
    csv_path = f"{save_prefix}_per_sample_predictions.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if config['task'] == "multi-label, binary-class":
            thr = float(config.get("fixed_threshold", 0.5))  # 固定阈值

            # ——统一表头：index / exact_match@thr / contains_all_true@thr / num_pos_true / num_pos_pred@thr / true:* / prob:* / pred@thr:*——
            header = (["index", f"exact_match@{thr:.2f}", f"contains_all_true@{thr:.2f}", "num_pos_true",
                       f"num_pos_pred@{thr:.2f}"]
                      + [f"true:{n}" for n in class_names]
                      + [f"prob:{n}" for n in class_names]
                      + [f"pred@{thr:.2f}:{n}" for n in class_names])
            writer.writerow(header)

            for i in range(y_true_np.shape[0]):
                true_row = y_true_np[i].astype(int).tolist()  # [C]
                prob_row = y_pred_np[i].astype(float).tolist()  # [C]
                pred_row = [int(p >= thr) for p in prob_row]  # [C]

                exact_match = int(true_row == pred_row)
                num_pos_true = int(sum(true_row))
                num_pos_pred = int(sum(pred_row))

                pos_idx = (y_true_np[i] == 1)  # [C] bool
                pred_row_arr = np.asarray(pred_row, dtype=int)  # [C] int
                contains_all_true = int(np.all(pred_row_arr[pos_idx] == 1)) if np.any(pos_idx) else 1

                writer.writerow([i, exact_match, contains_all_true, num_pos_true, num_pos_pred] +
                                true_row + prob_row + pred_row)

        else:
            # 多类互斥：index + true_cls + prob:每类 + pred_cls + 是否一致（保持原样）
            header = ["index", "true_class"] + [f"prob:{n}" for n in class_names] + ["pred_class", "match"]
            writer.writerow(header)

            for i in range(y_true_np.shape[0]):
                true_cls = int(y_true_np[i].ravel()[0])
                prob_row = y_pred_np[i].astype(float).tolist()
                pred_cls = int(np.argmax(prob_row))
                match = int(pred_cls == true_cls)
                writer.writerow([i, true_cls] + prob_row + [pred_cls, match])

    print(f"\t[OK] 已保存逐样本CSV到：{csv_path}")

    # ——新增：基线（无CP）每类正样本覆盖率，直接基于当前概率与 thr 计算并落盘，同时打印 overall 与每类覆盖率
    if config['task'] == "multi-label, binary-class":
        thr = float(config.get("fixed_threshold", 0.5))
        _save_per_class_threshold_coverage(class_names, y_true_np, y_pred_np, thr, save_prefix)

    # 5) 保存 JSONL（每行一个样本，便于后续脚本处理）
    jsonl_path = f"{save_prefix}_per_sample_predictions.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as jf:
        if config['task'] == "multi-label, binary-class":
            for i in range(y_true_np.shape[0]):
                rec = {
                    "index": i,
                    "true": y_true_np[i].astype(int).tolist(),
                    "prob": [float(v) for v in y_pred_np[i].tolist()],
                }
                jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
        else:
            for i in range(y_true_np.shape[0]):
                true_cls = int(y_true_np[i].ravel()[0])
                rec = {
                    "index": i,
                    "true_class": true_cls,
                    "prob": [float(v) for v in y_pred_np[i].tolist()],
                    "pred_class": int(np.argmax(y_pred_np[i])),
                }
                jf.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\t[OK] 已保存逐样本JSONL到：{jsonl_path}")

    # ======== 6) 若存在 CP 校准器，则追加计算/保存 CP 结果（表头同样统一） ========
    cp_file = config.get('cp_calibrator_file')
    if (config['task'] == "multi-label, binary-class") and cp_file and os.path.exists(cp_file):
        tau, alpha, cluster_map, q_per_cluster, q_global = _load_cp_calibrator(cp_file)
        if alpha is None:
            alpha = float(config.get('cp_alpha', 0.10))

        # 集合预测（CP）
        pred_bin_cp = (y_pred_np >= tau[None, :]).astype(int)    # [N, C]

        # 覆盖率（只在 y=1 上统计）
        pos_mask = (y_true_np == 1)
        covered = (pred_bin_cp[pos_mask] == 1).sum()
        totalpos = int(pos_mask.sum())
        pos_coverage = covered / max(totalpos, 1)

        # 平均集合大小 & Hamming-ACC@CP
        mean_set_size = pred_bin_cp.sum(axis=1).mean()
        hamming_acc_cp = 1.0 - np.mean((pred_bin_cp != y_true_np).astype(float))

        print(f"\t[CP] target_cov≈{1.0 - alpha:.3f} | pos_coverage={pos_coverage:.3f} | "
              f"mean_set_size={mean_set_size:.2f} | Hamming-ACC@CP={hamming_acc_cp:.3f}")

        # ——把上述打印信息也保存到一个新的 JSON 文件——
        cp_summary_path = f"{save_prefix}_cp_metrics_summary.json"
        cp_summary = {
            "target_cov": float(round(1.0 - alpha, 6)),
            "pos_coverage": float(round(pos_coverage, 6)),
            "mean_set_size": float(round(float(mean_set_size), 6)),
            "hamming_acc_cp": float(round(float(hamming_acc_cp), 6)),
            "alpha": float(alpha)
        }
        with open(cp_summary_path, "w", encoding="utf-8") as jf:
            json.dump(cp_summary, jf, ensure_ascii=False, indent=2)
        print(f"\t[CP] summary json saved to: {cp_summary_path}")

        # ——统一表头的 withCP CSV（包含 contains_all_true@CP）——
        csv_cp_path = f"{save_prefix}_per_sample_predictions_withCP.csv"
        with open(csv_cp_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            header_cp = (["index", "exact_match@CP", "contains_all_true@CP", "num_pos_true", "num_pos_pred@CP"]
                         + [f"true:{n}" for n in class_names]
                         + [f"prob:{n}" for n in class_names]
                         + [f"pred@CP:{n}" for n in class_names])
            writer.writerow(header_cp)

            for i in range(y_true_np.shape[0]):
                true_row = y_true_np[i].astype(int).tolist()
                prob_row = y_pred_np[i].astype(float).round(6).tolist()
                pred_cp_row = pred_bin_cp[i].astype(int).tolist()

                exact_match_cp = int(np.array_equal(pred_bin_cp[i], y_true_np[i]))
                num_pos_true = int(y_true_np[i].sum())
                num_pos_pred_cp = int(pred_bin_cp[i].sum())
                # 新增：是否“包含所有真正正项”（若无正项则定义为1）
                pos_idx = (y_true_np[i] == 1)
                contains_all_true = int(np.all(pred_bin_cp[i][pos_idx] == 1)) if np.any(pos_idx) else 1

                writer.writerow([i, exact_match_cp, contains_all_true, num_pos_true, num_pos_pred_cp] +
                                true_row + prob_row + pred_cp_row)

        print(f"\t[CP] per-sample with CP saved to: {csv_cp_path}")
        # 生成每类CP覆盖率报告（不读CSV，直接用当前内存中的数组计算，最快）
        _save_per_class_cp_coverage(class_names, y_true_np, pred_bin_cp, save_prefix)

    # ======== 保存每张图片的预测 END ========
