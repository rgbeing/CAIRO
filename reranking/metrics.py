import math


def ndcg_at_k(rank, k):
    """
    Compute nDCG@k for a single query with one relevant item.

    Args:
        rank (int): 1-indexed position of the ground-truth item in the ranked list.
        k (int): Cutoff rank.

    Returns:
        float: nDCG@k value. With a single relevant item, IDCG = 1/log2(2) = 1,
               so nDCG@k = 1/log2(rank+1) if rank <= k, else 0.
    """
    if rank <= k:
        return 1.0 / math.log2(rank + 1)
    return 0.0


def recall_at_k(rank, k):
    """
    Compute Recall@k (equivalent to Hit@k) for a single query with one relevant item.

    Args:
        rank (int): 1-indexed position of the ground-truth item.
        k (int): Cutoff rank.

    Returns:
        float: 1.0 if rank <= k, else 0.0.
    """
    return 1.0 if rank <= k else 0.0


def compute_metrics(gt_ranks, ks=(5, 10, 20)):
    """
    Compute averaged nDCG@k and Recall@k over all test samples.

    Args:
        gt_ranks (list[int]): 1-indexed rank of the ground-truth item for each test sample.
        ks (tuple[int]): Cutoff values.

    Returns:
        dict: Keys are 'nDCG@k' and 'Recall@k' for each k, values are averages.
    """
    n = len(gt_ranks)
    if n == 0:
        return {}

    results = {}
    for k in ks:
        results[f'nDCG@{k}']   = sum(ndcg_at_k(r, k)   for r in gt_ranks) / n
        results[f'Recall@{k}'] = sum(recall_at_k(r, k) for r in gt_ranks) / n

    return results
