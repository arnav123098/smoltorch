import torch

class Flatten:
    def __call__(self, input: torch.Tensor) -> torch.Tensor:
        self.out = input.view((input.shape[0], -1))
        return self.out
    
    def parameters(self): return []

class FlattenConsecutive:
    def __init__(
        self,
        n: int
    ):
        self.n = n

    def __call__(self, input: torch.Tensor) -> torch.Tensor:
        B, T, C = input.shape
        self.out = input.view(B, T//self.n, C*self.n)
        if self.out.shape[1] == 1:
            self.out = self.out.squeeze(1)
        return self.out
    
    def parameters(self): return []
