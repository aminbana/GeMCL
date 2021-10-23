from model.ProbModel import ProbModel
import torch.nn as nn
import torch
import torch.nn.functional as F
from model.Utils import weights_init


class MAP(ProbModel):
    def __init__(self, args):
        super(MAP, self).__init__(args)
        self.apply(weights_init)
        self.alpha = 100
        self.beta = 1000

    def classify(self, embedded, proto=None, var=None):
        if(proto is None):
            proto = self.prototypes
        if(var is None):
            var = self.var

        embedded = embedded.unsqueeze(1)
        diff = (embedded - proto.unsqueeze(0))**2
        t1 = -(diff / (2*var.unsqueeze(0))).sum(dim=-1)
        t2 = -(var.log()/2).sum(dim=-1).reshape(1, -1)
        return t1 + t2
