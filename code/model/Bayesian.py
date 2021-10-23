from model.ProbModel import ProbModel
import torch.nn as nn
import torch
import torch.nn.functional as F
from model.Utils import weights_init

class Bayesian(ProbModel):
    def __init__(self, args):
        super(Bayesian, self).__init__(args)
        self.apply(weights_init)
        self.alpha = 100
        self.beta = 1000

    def marginals(self, embedded, proto):
        beta = (self.beta_hat**2 + 1e-5)
        alpha = (self.alpha_hat**2 + 1e-5)

        d = self.embedding_size
        z = (beta.log()* alpha) * d
        z = z - torch.lgamma(alpha) * d
        alpha_p = alpha + 0.5
        z = z + torch.lgamma(alpha_p) * d
        
        embedded = embedded.unsqueeze(1)
        diff = (embedded - proto.unsqueeze(0))**2
        beta_p = beta + diff/2
        z = z - beta_p.log().sum(dim=-1) * alpha_p
        return z

    def predictive(self, embedded, proto):
        beta = self.beta_prime
        alpha = self.alpha_prime
        
        z = (beta.log() * alpha).sum(dim=-1)
        z = z - torch.lgamma(alpha).sum(dim=-1)
        alpha_p = alpha + 0.5
        z = z + torch.lgamma(alpha_p).sum(dim=-1)

        embedded = embedded.unsqueeze(1)
        diff = (embedded - proto.unsqueeze(0))**2
        beta_p = beta.unsqueeze(0) + diff/2
        z = z - (beta_p.log() * alpha_p.unsqueeze(0)).sum(dim=-1)
        return z

    def classify(self, embedded, proto=None, var=None):
        if(proto is None):
            proto = self.prototypes
        assert var is None, "Bayesian model doesnt need vars"
        return self.predictive(embedded, proto)

