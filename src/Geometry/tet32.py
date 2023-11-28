import torch

## Implements the 32 bits tetrahedral mesh data structure
class Tet32:
    def __init__(self, nb_tets):
        self.device = torch.device('cuda')

        ## 4 values for indices of summits
        self.summits = torch.zeros([nb_tets, 4], dtype = torch.int32).cuda().contiguous()

        ## 4 values for indices of neighbors
        self.neighbors = torch.zeros([nb_tets, 4], dtype = torch.int32).cuda().contiguous()