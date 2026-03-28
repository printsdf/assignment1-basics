import math
import torch


class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        weight = torch.empty(
            (out_features, in_features),
            device=device,
            dtype=dtype,
        )
        std = math.sqrt(2.0 / (in_features + out_features))
        torch.nn.init.trunc_normal_(weight, mean=0.0, std=std, a=-3 * std, b=3 * std)
        self.weight = torch.nn.Parameter(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T

