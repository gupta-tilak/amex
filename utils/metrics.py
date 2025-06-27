import numpy as np
import pandas as pd

def mapk(actual, predicted, k=7):
    """
    Computes the mean average precision at k (MAP@k).
    actual: list of lists (true clicked offers per user)
    predicted: list of lists (top-k predicted offers per user)
    k: int, number of top predictions to consider
    """
    def apk(a, p, k):
        if len(p) > k:
            p = p[:k]
        score = 0.0
        num_hits = 0.0
        for i, pred in enumerate(p):
            if pred in a and pred not in p[:i]:
                num_hits += 1.0
                score += num_hits / (i + 1.0)
        if not a:
            return 0.0
        return score / min(len(a), k)
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])

def map7_from_dataframe(df, id_col='id2', offer_col='id3', label_col='y', pred_col='pred', k=7):
    """
    Computes MAP@7 from a DataFrame with columns for user, offer, true label, and predicted score.
    - df: DataFrame with columns [id_col, offer_col, label_col, pred_col]
    - id_col: user/customer ID
    - offer_col: offer ID
    - label_col: true label (1 if clicked, 0 otherwise)
    - pred_col: predicted probability or score
    """
    grouped = df.groupby(id_col)
    actual = grouped.apply(lambda x: x.loc[x[label_col] == 1, offer_col].tolist()).tolist()
    predicted = grouped.apply(lambda x: x.sort_values(pred_col, ascending=False)[offer_col].tolist()[:k]).tolist()
    return mapk(actual, predicted, k) 