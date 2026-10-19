import math


def dcg(relevances: list[float]) -> float:
    return sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(relevances))


def ndcg_at_k(ranked_ids: list[str], labels: dict[str, float], k: int) -> float:
    gains = [labels.get(i, 0) for i in ranked_ids[:k]]
    ideal = sorted(labels.values(), reverse=True)[:k]
    best = dcg(ideal)
    return dcg(gains) / best if best else 0.0


def precision_at_k(ranked_ids: list[str], labels: dict[str, float], k: int, min_rel: float = 2) -> float:
    return sum(labels.get(i, 0) >= min_rel for i in ranked_ids[:k]) / k
