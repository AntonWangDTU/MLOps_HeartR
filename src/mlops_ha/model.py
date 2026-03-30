from torch import nn
import torch


class Model(nn.Module):
    """
    Feedforward neural network for classification.
    Works for binary and multi-class classification.
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: list[int] = [64, 32],
        output_size: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()

        layers = []
        in_features = input_size

        # Hidden layers
        for h in hidden_sizes:
            layers.append(nn.Linear(in_features, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_features = h

        # Output layer
        layers.append(nn.Linear(in_features, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.network(x)

        # Binary classification
        if logits.shape[-1] == 1:
            return torch.sigmoid(logits)

        # Multi-class classification
        return torch.softmax(logits, dim=1)


if __name__ == "__main__":
    model = Model(input_size=13, output_size=1)  # e.g. heart dataset
    x = torch.rand(5, 13)
    print(f"Output shape of model: {model(x).shape}")
