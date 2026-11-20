import torch
import torch.nn.functional as F

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> tuple[list[float], float]:
    """
    Simulates a single neuron with sigmoid activation for binary classification.
    
    Args:
        features: List of feature vectors (each a list of floats)
        labels: List of true binary labels
        weights: Neuron weights (one per feature)
        bias: Neuron bias term
    
    Returns:
        Tuple of (predicted probabilities rounded to 4 decimal places, MSE rounded to 4 decimal places)
    """
    # Your code here using PyTorch built-ins:
    # - torch.matmul() for linear combination
    # - torch.sigmoid() for activation
    # - torch.nn.functional.mse_loss() for MSE
    X=torch.tensor(features, dtype=torch.float32)
    W=torch.tensor(weights, dtype=torch.float32)
    Y=torch.tensor(labels, dtype=torch.float32)
    b=bias
    Z=torch.matmul(X,W)+b
    Y_p=torch.sigmoid(Z)
    MSE=torch.nn.functional.mse_loss(Y,Y_p)
    return[round(x,4)for x in Y_p.tolist()],round(MSE.item(),4)

