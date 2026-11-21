import torch
import torch.nn as nn

def train_neuron(features: torch.Tensor, labels: torch.Tensor, initial_weights: torch.Tensor, initial_bias: float, learning_rate: float, epochs: int) -> tuple[list[float], float, list[float]]:
    """
    Simulates a single neuron with sigmoid activation and trains it using
    backpropagation with MSE loss via SGD.

    Args:
        features: Input feature tensor of shape (n_samples, n_features)
        labels: Binary label tensor of shape (n_samples,)
        initial_weights: Initial weight tensor of shape (n_features,)
        initial_bias: Initial bias scalar
        learning_rate: Learning rate for SGD
        epochs: Number of training epochs

    Returns:
        Tuple of (updated_weights, updated_bias, mse_values) all rounded to 4 decimal places
    """
    X=torch.tensor(features, dtype=torch.float32)
    W=torch.tensor(initial_weights,  dtype=torch.float32, requires_grad=True)
    Y=torch.tensor(labels,  dtype=torch.float32)
    b=torch.tensor(float(initial_bias), dtype=torch.float32, requires_grad=True)
    lr=learning_rate
    optimizer=torch.optim.SGD([W,b],lr=lr)
    mse_values=[]
    def train(X,W,Y,B, optimizer):
      optimizer.zero_grad()
      Z=torch.matmul(X,W)+b
      Y_P=torch.sigmoid(Z)
      loss_mse=torch.nn.functional.mse_loss(Y,Y_P)
      loss_mse.backward()
      optimizer.step()
      return loss_mse.item()
    for epoch in range (epochs):
        loss=train(X,W,Y,b, optimizer)
        updates_weights=[round(w,4) for w in W.tolist()]
        updated_bias=round(b.item(),4)
        mse_values.append(round(loss,4))
    return updates_weights, updated_bias, mse_values

