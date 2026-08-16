import math


def compute_ppl(loss: float) -> float:
    if loss > 20:
        return float("inf")
    return math.exp(loss)