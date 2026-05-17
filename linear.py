import torch
import torch.nn.functional as F

class Linear:
    def __init__(
            self,
            fan_in: int,
            fan_out: int,
            bias: bool = True,
            generator: torch.Generator = None
        ):
        self.weights = torch.randn((fan_out, fan_in), generator=generator)
        self.bias = torch.randn(fan_out, generator=generator) if bias else None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weights, self.bias)
    
    def parameters(self) -> list[torch.Tensor]:
        return [self.weights] + ([] if self.bias is not None else self.bias)
