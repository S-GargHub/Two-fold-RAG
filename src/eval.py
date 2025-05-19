import util

def exact_match(pred, gold):
    return util.clean(pred) == util.clean(gold)

def context_precision(retrieved_context, ground_truth):
    relevant = [c for c in retrieved_context if util.clean(ground_truth) in util.clean(c)]
    return len(relevant) / len(retrieved_context) if retrieved_context else 0

def context_recall(retrieved_context, ground_truth):
    return int(any(util.clean(ground_truth) in util.clean(c) for c in retrieved_context))

def top_k_accuracy(retrieved_context, ground_truth, k=5):
    top_k = retrieved_context[:k]
    return int(any(util.clean(ground_truth) in util.clean(c) for c in top_k))

def kg_hit_rate(result):
    return int(result == 'KG')
