import torch

## Implements the 32 bits tetrahedral mesh data structure
class Tet20:
    def __init__(self):
        self.device = torch.device('cuda')