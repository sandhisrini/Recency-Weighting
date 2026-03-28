import numpy as np


def score1(rule, c=3, weight = 1.0, weight_constraint = 0.5, weight_no_constraint = 0.5):



    #score = rule["rule_supp"] / (rule["body_supp"] + c)
    if not rule["acyclic"]:
        score = weight * rule["rule_supp"] / (rule["body_supp"] + c)
    else:
        if rule["no_constraint"]:
            score = weight_no_constraint * rule["rule_supp"] / (rule["body_supp"] + c)
        else:
            score = weight_constraint * rule["rule_supp"] / (rule["body_supp"] + c)

    return score


def score2(cands_walks, test_query_ts, lmbda):


    max_cands_ts = max(cands_walks["timestamp_0"])
    score = np.exp(
        lmbda * (max_cands_ts - test_query_ts)
    )  # Score depending on time difference

    return score


def score_12(rule, cands_walks, test_query_ts, lmbda, a):
    """
    Combined score (confidence + time decay).
    Equation (3) from TR-Rules paper:
    f(R, c) = a * conf(R) + (1 - a) * exp(-λ(t - t_1(B(R, c))))
    """
    base_score = a * score1(rule) + (1 - a) * score2(cands_walks, test_query_ts, lmbda)
    return base_score

def score_12_variant_a(rule, cands_walks, test_query_ts, lmbda, a, recency_weight=0.3):
    """
    VARIANT A: Recency-weighted scoring
    
    Weight body instances by recency: closer instances have higher weight.
    Rationale: More recent body instances are more predictive.
    
    Formula: f(R, c) = a * conf(R) + (1 - a) * exp(-λ * time_diff * recency_penalty)
    where recency_penalty = 1 + (1 - recency_factor)
    """
    base_conf = a * score1(rule)
    
    # Get all timestamps from body instances
    try:
        timestamps = np.array(cands_walks["timestamp_0"])
        if len(timestamps) == 0:
            return base_conf
            
        # Recency factor: newer = closer to test_query_ts = higher weight
        time_diffs = test_query_ts - timestamps
        recency_scores = np.exp(-recency_weight * time_diffs)
        avg_recency = np.mean(recency_scores)
        
        # Time decay component with recency boost
        max_cands_ts = max(timestamps)
        time_decay = np.exp(lmbda * (max_cands_ts - test_query_ts))
        
        return base_conf + (1 - a) * time_decay * (1 + avg_recency)
    except:
        return a * score1(rule) + (1 - a) * score2(cands_walks, test_query_ts, lmbda)


def score_12_variant_b(rule, cands_walks, test_query_ts, lmbda, a, acyclic_penalty=0.7):
    """
    VARIANT B: Acyclic rule penalty
    
    Reduce weight of acyclic rules since they're noisier and benefit more
    from window confidence. Apply confidence penalty if acyclic.
    
    Rationale: Acyclic rules are more sensitive to temporal redundancy,
    so reduce their contribution.
    """
    base_score = a * score1(rule) + (1 - a) * score2(cands_walks, test_query_ts, lmbda)
    
    # If acyclic, apply penalty to confidence component only
    if rule.get("acyclic", False):
        conf = score1(rule)
        time_decay = score2(cands_walks, test_query_ts, lmbda)
        # Penalize acyclic confidence
        penalized_conf = a * (conf * acyclic_penalty) + (1 - a) * time_decay
        return penalized_conf
    
    return base_score


def score_12_variant_c(rule, cands_walks, test_query_ts, lmbda, a, window_smoothing=0.1):
    """
    VARIANT C: Aggregate body instances more carefully
    
    When multiple body instances exist, smooth their contribution using
    normalized count: count / (1 + count) instead of raw count.
    
    Rationale: Avoids over-weighting when there are many body instances.
    This is applied IN the window confidence, not after scoring.
    """
    base_score = a * score1(rule) + (1 - a) * score2(cands_walks, test_query_ts, lmbda)
    
    try:
        count = len(cands_walks)
        # Smooth aggregation: curve that grows but plateaus
        # At count=1: 0.5, count=2: 0.67, count=5: 0.83, count=10: 0.91
        smoothed = count / (count + 1 + window_smoothing)
        return base_score * (1 + smoothed)  # Modest boost, not logarithmic
    except:
        return base_score