# cp/clustered_cp.py
import numpy as np
from math import ceil
from typing import List, Dict, Tuple, Optional
from sklearn.cluster import KMeans
import torch
from torch.utils.data import DataLoader

# ===========================
# 基本工具
# ===========================
def _finite_sample_quantile(vec: np.ndarray, alpha: float) -> float:    #样本分位数
    """
    split-conformal 有限样本分位数：
      k = ceil((n+1)*(1-alpha))
      q = 第 k 小的值（从 1 开始）
    """
    n = len(vec)
    if n == 0:
        return np.nan
    k = int(ceil((n + 1) * (1 - alpha)))
    k = min(max(k, 1), n)
    arr = np.sort(vec)
    return float(arr[k - 1])

@torch.no_grad()
def _collect_positive_scores(   #非一致性分数
    model: torch.nn.Module,
    device: torch.device,
    loader: DataLoader
) -> List[np.ndarray]:
    """
    在给定 loader 上前向，收集“正例(y=1)的非一致性分数 s=1-p”。
    返回：长度为 C 的 list，第 j 个元素是该类在此集合里的正例 s 向量（可能为空）。
    说明：本函数**仅用于多标签**任务；模型输出 logits（未过sigmoid）。
    """
    model.eval()
    scores_by_class = None  # list of list[np.ndarray]
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.cpu().numpy().astype(int)        # [B, C]（B=batch size，C=类别数）
        logits = model(images)                           # [B, C]
        probs  = torch.sigmoid(logits).cpu().numpy()     # [B, C]
        s      = 1.0 - probs                             # 非一致性分数(正例越小越好)
        B, C = s.shape
        if scores_by_class is None:
            scores_by_class = [[] for _ in range(C)]
        for j in range(C):
            pos_mask = (labels[:, j] == 1)
            if np.any(pos_mask):
                scores_by_class[j].append(s[pos_mask, j])

    # 拼接
    out = []
    for buf in scores_by_class:
        if len(buf) == 0:
            out.append(np.empty((0,), dtype=float))
        else:
            out.append(np.concatenate(buf, axis=0).astype(float))
    return out  # len C, each 1D array

def _quantile_embedding_with_fs_adjust(
    scores_pos: np.ndarray,
    alpha: float,
    base_taus: Tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9),
) -> Optional[np.ndarray]:
    """
    按原文做“分位数嵌入”：
      T = {0.5,0.6,0.7,0.8,0.9} ∪ {1-α}，并对每个类用 n=|I_1^y| 做有限样本调整：
      τ_eff = ceil((n+1)*τ)/n，再以“最近阶”经验分位数取值。
    返回 z ∈ R^{|T|}；若该类无正例返回 None。
    """
    n = int(scores_pos.size)
    if n == 0:
        return None
    # 组装 T 并去重排序
    taus = sorted(set(list(base_taus) + [1.0 - float(alpha)]))
    # 有限样本调整（落在 [1/n, 1]）
    taus_eff = np.array([min(max(np.ceil((n + 1) * t) / n, 1.0 / n), 1.0) for t in taus], dtype=float)
    # “最近阶”经验分位数（兼容旧 numpy）
    try:
        z = np.quantile(scores_pos, q=taus_eff, method="nearest")
    except TypeError:
        z = np.quantile(scores_pos, q=taus_eff, interpolation="nearest")
    return z.astype(float)

# ===========================
# 聚类 CP 主函数
# ===========================
@torch.no_grad()
def fit_clustered_cp_return_classwise_tau(
    model: torch.nn.Module,
    device: torch.device,
    D1_loader: DataLoader,
    D2_loader: DataLoader,
    *,
    alpha: float = 0.10,
    # 注意：这里的参数代表“基础点 base_taus”，最终会与 {1-α} 合并后再做有限样本调整
    taus_for_embed: Tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9),
    n_clusters: int = 4,
    min_pos_for_embed: Optional[int] = None,
    random_state: int = 2025,
) -> Dict[str, np.ndarray]:
    """
    使用 D1/D2 构造“类共享（聚类）CP”，并返回**每类阈值向量** tau_per_class（长度 C）。

    输入：
      - model: 已加载 best.pth 的模型（eval 模式）
      - device: 设备
      - D1_loader: 仅用于聚类“类”的集合（val 的子集）
      - D2_loader: 仅用于估分位数阈值的集合（val 的子集，与 D1 不重叠）
      - alpha: 失效水平（覆盖目标约为 1-alpha）
      - taus_for_embed: 基础点 T_base（默认 {0.5,0.6,0.7,0.8,0.9}），实际用 T = T_base ∪ {1-α}
      - n_clusters: KMeans 的簇数（若有效类少，会自动下调）
      - min_pos_for_embed: 把某类纳入聚类需满足的最少正例数；
                           默认为 n_α = (1/min(alpha,0.1)) - 1（α=0.1→9, α=0.05→19）
      - random_state: 复现用随机种子

    返回：
      {
        "tau_per_class": (C,) 每类概率阈值（用于 p_j >= tau[j] 规则），含回退策略；
        "cluster_map":   (C,) 类 j -> 簇 id（-1 表示稀有类/未聚类，走全局回退）；
        "q_per_cluster": (M,) 每簇 (1-α) 的 s 分位数；可能含 NaN（无正例时）；
        "q_global":      标量，全局回退的 (1-α) s 分位数；可能为 NaN（极端无正例）；
        "alpha":         标量，记录 alpha；
        "taus_for_embed":(len(T),) 嵌入点位（基础点 + 1-α，经有限样本调整前的原始点位）；
      }
    """
    model.eval()

    # ---- 1) D1：为每个“类”构建分位数嵌入 → 聚“类” ----
    scores_D1 = _collect_positive_scores(model, device, D1_loader)  # list len C
    C = len(scores_D1)
    if min_pos_for_embed is None:
        min_pos_for_embed = int((1 / max(alpha, 0.1)) - 1) #稀有类门槛数量

    Z, weights, valid_class_idx = [], [], []
    for j in range(C):
        n_pos = scores_D1[j].size   #统计类 j 在 D1 里的正例数
        if n_pos >= min_pos_for_embed:
            z = _quantile_embedding_with_fs_adjust(scores_D1[j], alpha=alpha, base_taus=taus_for_embed)
            if z is not None:
                Z.append(z)
                weights.append(np.sqrt(float(n_pos)))  # |I^1_y|^{1/2}
                valid_class_idx.append(j)

    if len(valid_class_idx) == 0:
        # 极端情况：D1 中没有任何类满足门槛，直接退化为“全局阈值”
        scores_D2 = _collect_positive_scores(model, device, D2_loader)
        pool = np.concatenate([s for s in scores_D2 if s.size > 0], axis=0) if any(s.size>0 for s in scores_D2) else np.empty((0,))
        q_global = _finite_sample_quantile(pool, alpha) if pool.size>0 else np.nan
        tau_global = 1.0 - q_global if not np.isnan(q_global) else 0.5
        return dict(
            tau_per_class=np.full(C, tau_global, dtype=float),
            cluster_map=np.full(C, -1, dtype=int),
            q_per_cluster=np.empty((0,), dtype=float),
            q_global=q_global,
            alpha=float(alpha),
            taus_for_embed=np.array(sorted(set(list(taus_for_embed) + [1.0 - float(alpha)])), dtype=float),
        )

    Z = np.stack(Z, axis=0)                      # [K_eff, |T|]
    weights = np.asarray(weights, dtype=float)   # [K_eff]

    M = min(n_clusters, len(valid_class_idx))
    kmeans = KMeans(n_clusters=M, n_init="auto", random_state=random_state)
    kmeans.fit(Z, sample_weight=weights)

    cluster_map = np.full(C, -1, dtype=int)     # 默认 -1：稀有类/不参与聚类
    for lab, j in zip(kmeans.labels_, valid_class_idx):
        cluster_map[j] = int(lab)

    # ---- 2) D2：按“簇”汇总正例 s，再取 (1-α) 分位数 → 得到 q̂_簇 ----
    scores_D2 = _collect_positive_scores(model, device, D2_loader)
    pools = [ [] for _ in range(M) ]
    global_pool = []
    for j in range(C):
        s = scores_D2[j]
        if s.size == 0:
            continue
        global_pool.append(s)
        c = cluster_map[j]
        if c >= 0:
            pools[c].append(s)

    q_per_cluster = np.full(M, np.nan, dtype=float)
    for c in range(M):
        if len(pools[c]) > 0:
            vec = np.concatenate(pools[c], axis=0)
            q_per_cluster[c] = _finite_sample_quantile(vec, alpha)

    q_global = _finite_sample_quantile(np.concatenate(global_pool, axis=0), alpha) if len(global_pool)>0 else np.nan

    # ---- 3) 生成“每类概率阈值向量” tau_per_class ----
    # 规则：类 j 属于簇 c，则 tau[j] = 1 - q_per_cluster[c]；
    #      若 c=-1 或该簇 q 为 NaN，则回退 tau[j] = 1 - q_global；
    fallback_tau = 1.0 - q_global if not np.isnan(q_global) else 0.5
    tau_per_class = np.full(C, fallback_tau, dtype=float)
    for j in range(C):
        c = cluster_map[j]
        if c >= 0 and not np.isnan(q_per_cluster[c]):
            tau_per_class[j] = 1.0 - q_per_cluster[c]

    return dict(
        tau_per_class=tau_per_class,
        cluster_map=cluster_map,
        q_per_cluster=q_per_cluster,
        q_global=q_global,
        alpha=float(alpha),
        taus_for_embed=np.array(sorted(set(list(taus_for_embed) + [1.0 - float(alpha)])), dtype=float),
    )
