import torch

class BatchNorm1d: # B, C, T
    def __init__(
            self,
            dim: int,
            eps: float = 1e-05,
            momentum: float = 0.1
        ):
        self.eps = eps
        self.momentum = momentum
        self.training = True

        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)

        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    def __call__(self, input: torch.Tensor) -> torch.Tensor:
        assert input.ndim in (2, 3), "Only supports 2D or 3D inputs"
        dims = (0, 2) if input.ndim == 3 else 0
        shape = (1, -1, 1) if input.ndim == 3 else (-1)

        if self.training:
            mean = input.mean(dims, keepdim=True)
            var = input.var(dims, keepdim=True)
        else:
            mean = self.running_mean
            var = self.running_var

        g = self.gamma.view(shape)
        b = self.beta.view(shape)

        xhat = (input - mean) / torch.sqrt(var + self.eps)
        self.out = g * xhat + b

        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + (self.momentum * mean)
                self.running_var = (1 - self.momentum) * self.running_var + (self.momentum * var)

        return self.out
    
    def parameters(self):
        return [self.gamma, self.beta]

class LayerNorm: # B, T, C
    def __init__(
            self,
            dim: int,
            eps: float = 1e-05
    ):
        self.eps = eps

        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)

    def __call__(self, input):
        mean = input.mean(-1, keepdim=True)
        var = input.var(-1, keepdim=True)

        xhat = (input - mean) / torch.sqrt(var + self.eps)
        self.out = self.gamma * xhat + self.beta

        return self.out
    
    def parameters(self):
        return [self.gamma, self.beta]
