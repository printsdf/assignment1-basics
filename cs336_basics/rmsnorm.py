import torch

class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        weight = torch.ones(
            d_model,
            device=device,
            dtype=dtype
        )
        self.weight = torch.nn.Parameter(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype= x.dtype
        x = x.to(torch.float32)

        rms = torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True)) + self.eps
        out = (x / rms) * self.weight

        return out.to(in_dtype)