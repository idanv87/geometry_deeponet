import torch
import numpy as np

class norms:
    def __init__(self): 
        pass
    @classmethod
    def relative_L2(cls,x,y):
        return torch.norm(x-y)/(torch.norm(y)+1e-10)