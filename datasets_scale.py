# [NPZ] C:\Users\WANG\Desktop\rethinking-model-prototyping-MedMNISTPlus-main\data\chestmnist.npz
#   train_labels: shape=(78468, 14), dtype=uint8
#     train 每个标签的图片数（阳性计数）:
#     0:atelectasis=7996, 1:cardiomegaly=1950, 2:effusion=9261, 3:infiltration=13914, 4:mass=3988, 5:nodule=4375, 6:pneumonia=978,
#     7:pneumothorax=3705, 8:consolidation=3263, 9:edema=1690, 10:emphysema=1799, 11:fibrosis=1158, 12:pleural=2279, 13:hernia=144
#     train 样本总数: 78468
#   val_labels: shape=(11219, 14), dtype=uint8
#     val 每个标签的图片数（阳性计数）:
#     0:atelectasis=1119, 1:cardiomegaly=240, 2:effusion=1292, 3:infiltration=2018, 4:mass=625, 5:nodule=613, 6:pneumonia=133,
#     7:pneumothorax=504, 8:consolidation=447, 9:edema=200, 10:emphysema=208, 11:fibrosis=166, 12:pleural=372, 13:hernia=41
#     val 样本总数: 11219
#   test_labels: shape=(22433, 14), dtype=uint8
#     test 每个标签的图片数（阳性计数）:
#     0:atelectasis=2420, 1:cardiomegaly=582, 2:effusion=2754, 3:infiltration=3938, 4:mass=1133, 5:nodule=1335, 6:pneumonia=242,
#     7:pneumothorax=1089, 8:consolidation=957, 9:edema=413, 10:emphysema=509, 11:fibrosis=362, 12:pleural=734, 13:hernia=42
#     test 样本总数: 22433
#
# [总计]
#   每个标签/类别的图片数:
#   0:atelectasis=11535, 1:cardiomegaly=2772, 2:effusion=13307, 3:infiltration=19870, 4:mass=5746, 5:nodule=6323, 6:pneumonia=1353,
#   7:pneumothorax=5298, 8:consolidation=4667, 9:edema=2303, 10:emphysema=2516, 11:fibrosis=1686, 12:pleural=3385, 13:hernia=227
#   样本总数: 112120
import numpy as np
from pathlib import Path
from medmnist import INFO

def count_per_label(npz_path: str, dataset_name: str):
    info = INFO[dataset_name]
    label_map = {int(k): v for k, v in info['label'].items()}
    num_labels = len(label_map)

    def names_line(counts):
        return ", ".join([f"{i}:{label_map[i]}={int(counts[i])}" for i in range(num_labels)])

    with np.load(npz_path, mmap_mode="r", allow_pickle=False) as f:
        print(f"[NPZ] {npz_path}")
        splits = ["train", "val", "test"]
        total_pos = np.zeros(num_labels, dtype=int)
        total_n = 0

        for sp in splits:
            key = f"{sp}_labels"
            if key not in f.files:
                continue
            y = f[key].squeeze()
            print(f"  {key}: shape={y.shape}, dtype={y.dtype}")

            # 多标签：形状 (N, L)；单标签：形状 (N,) 或 (N,1)
            if y.ndim == 2 and y.shape[1] > 1:
                # 多标签：每列求和 = 该标签出现的图片数
                pos = y.sum(axis=0).astype(int)                # (L,)
                n = y.shape[0]
                print(f"    {sp} 每个标签的图片数（阳性计数）: {names_line(pos)}")
                print(f"    {sp} 样本总数: {n}")
                total_pos += pos
                total_n += n
            else:
                # 单标签：类别ID计数
                cls = y.reshape(-1).astype(int)
                counts = np.bincount(cls, minlength=num_labels)
                n = cls.shape[0]
                print(f"    {sp} 每个类别的图片数: {names_line(counts)}")
                print(f"    {sp} 样本总数: {n}")
                total_pos += counts
                total_n += n

        print("\n[总计]")
        print("  每个标签/类别的图片数:", names_line(total_pos))
        print("  样本总数:", total_n)
        if (npz_path.lower().endswith("chestmnist.npz")):
            print("  注意：ChestMNIST 是多标签任务，同一张图片可能属于多个标签，上面各标签计数相加会大于样本总数。")

# 例子：把你的路径填进去
count_per_label(
    r"C:\Users\WANG\Desktop\rethinking-model-prototyping-MedMNISTPlus-main\data\chestmnist.npz",
    dataset_name="chestmnist"
)
