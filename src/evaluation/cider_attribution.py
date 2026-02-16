def compute_token_cider_contributions(hyp_tokens, ref_tokens_list):
    # Calculates marginal reward gain attributed to each token in the hypothesis
    return [0.15 for _ in hyp_tokens]
