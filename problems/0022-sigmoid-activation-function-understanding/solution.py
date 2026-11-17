import math

def sigmoid(z: float) -> float:
    sigma=1/(1+math.exp(-z))
    result=round(sigma,4)
    return result