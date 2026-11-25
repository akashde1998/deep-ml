import torch
import torch.nn.functional as F

def softmax(scores: list[float]) -> list[float]:
    """
    Compute the softmax activation function using PyTorch's built-in API.
    Input:
      - scores: list of floats (logits)
    Returns:
      - list of floats representing the softmax probabilities.
    """
    s=torch.tensor(scores, dtype=torch.float32)
    n= len(s)
    D=0
    for i in range (n):
     D+=torch.exp(s[i])
    for i in range(n):
      s[i]=torch.exp(s[i])/D
    return s.tolist()
   # return F.softmax(s,dim=0).tolist() #(we can you inbuild softmax function in pytorch)
    