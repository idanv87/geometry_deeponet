import torch
import numpy as np

class norms:
    def __init__(self): 
        pass
    @classmethod
    def relative_L2(cls,x,y):
        return torch.linalg.norm(x-y)/(torch.linalg.norm(y)+1e-10)
    @classmethod
    def relative_L1(cls,x,y):
        return torch.nn.L1Loss()(x,y)/(torch.nn.L1Loss(y,y*0)+1e-10)