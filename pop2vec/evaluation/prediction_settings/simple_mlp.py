import torch.nn as nn

class SimpleMLP(nn.Module):
  """Simple MLP with either 1 or 2 layers, depending on config."""

  def __init__(self, input_dim, output_dim, num_layers, activation_fn = None, dropout_rate = 0.1):
    """
    Args:
      input_dim: int, dimension of input features.
      output_dim: int, dimension of the output layer.
      num_layers: int, number of layers (1 or 2).
      activation_fn: str, name of the PyTorch activation function (e.g. 'ReLU').
    """
    super().__init__()
    self.num_layers = num_layers
    if num_layers == 1:
      self.net = nn.Linear(input_dim, output_dim)
    else:
      self.activation = getattr(nn, activation_fn, nn.ReLU)()
      if isinstance(self.activation, nn.modules.activation.ReLU) and activation_fn != 'ReLU':
        print(f"passed activation_fn {activation_fn} is invalid, using ReLU instead")
    
      self.lin1 = nn.Linear(input_dim, input_dim)
      self.drop1 = nn.Dropout(dropout_rate)
      self.lin2 = nn.Linear(input_dim, output_dim)
      self.drop2 = nn.Dropout(dropout_rate)

  def forward(self, x):
    """Forward pass of the network."""
    if self.num_layers == 1:
      return self.net(x)
    else:
      x = self.drop1(x)
      x = self.lin1(x)
      x = self.activation(x)
      x = self.drop2(x)
      x = self.lin2(x)
      return x
