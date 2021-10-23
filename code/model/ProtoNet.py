from model.BaseModel import BaseModel
import torch.nn as nn
import torch
import torch.nn.functional as F

class ProtoNet(BaseModel):
    def __init__(self, args):
        super(ProtoNet, self).__init__(args)

    def reset_episode(self, labels:torch.Tensor):
        self.prototypes = torch.empty((labels.shape[0], self.embedding_size), device=labels.device)

    def update_prototype(self, x: torch.Tensor, y:int, embed=True):
        self.shots = x.shape[0]
        if(embed):
            embedded = self.back_bone(x)
        else:
            embedded = x
        proto = embedded.mean(axis=0)
        self.prototypes[y] = proto



              

