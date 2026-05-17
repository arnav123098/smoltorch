import torch

class Embedding:
    def __init__(
        self,
        num_emb: int,
        emb_dim: int,
        generator: torch.Generator = None
    ):
        self.emb = torch.randn((num_emb, emb_dim), generator=generator)

    def __call__(self, input):
        self.out = self.emb[input]
        return self.out
    
    def parameters(self):
        return [self.emb]
