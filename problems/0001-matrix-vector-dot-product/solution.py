import torch

def matrix_dot_vector(a, b) -> torch.Tensor:
    """
    Compute the product of matrix `a` and vector `b` using PyTorch.
    Inputs can be Python lists, NumPy arrays, or torch Tensors.
    Returns a 1-D tensor of length m, or tensor(-1) if dimensions mismatch.
    """
    a=torch.as_tensor(a)
    b=torch.as_tensor(b)
    x=a.shape
    y=b.shape
    if x[1]!=y[0]:
        return torch.tensor(-1)
    else:
        return (torch.matmul(a,b))
    pass
