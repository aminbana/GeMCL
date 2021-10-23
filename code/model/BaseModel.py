import torch.nn as nn
import torch
import torch.nn.functional as F
from model.Utils import weights_init


class BaseModel(nn.Module):
    def __init__(self, args):
        super(BaseModel, self).__init__()
        
        self.args = args
        self.back_bone = args.backBone(input_channels = args.input_size[0])
        self.embedding_size = self.back_bone.get_embedding_size(input_size=args.input_size)
        self.shots = 0
        self.apply(weights_init)

    def reset_episode(self, labels:torch.Tensor):
        pass

    def update_prototype(self, x: torch.Tensor, y:int, embed=True):
        pass

    def prepare_prototypes(self):
        pass

    def classify(self, embedded, proto=None):
        pass
              
    def forward(self, x: torch.Tensor, y=None, embed=True):
        temp = self.args.temperature
        if(self.args.temperature is None):
            temp = 1
        
        if(embed):
            embedded = self.back_bone(x)
        else:
            embedded = x
        scores = self.classify(embedded) / temp
        return scores
        
    def classify(self, embedded, proto=None):
        if(proto is None):
            proto = self.prototypes

        if self.args.dist == 'euc':
            embedded = embedded.unsqueeze(1)
            return -((embedded - proto.unsqueeze(0))**2).sum(dim=-1).reshape(embedded.shape[0], proto.shape[0])
        if self.args.dist == 'cosine':
            embedded = embedded / torch.clamp(embedded.norm(dim=-1, keepdim=True), 1e-8)
            proto = proto / torch.clamp(proto.norm(dim=-1, keepdim=True), 1e-8)
            return  torch.mm(embedded, proto.T)
        if self.args.dist == 'norm-dot':
            proto = F.normalize(proto, dim=-1)
            return  torch.mm(embedded, proto.T)
        if self.args.dist == 'dot':
            return  torch.mm(embedded, proto.T)
        assert False 