from model.BaseModel import BaseModel
import torch.nn as nn
import torch
import torch.nn.functional as F
from model.Utils import weights_init


class ProbModel(BaseModel):
    def __init__(self, args):
        super(ProbModel, self).__init__(args)
        self.alpha = 100
        self.beta = 1000

    def reset_episode(self, labels:torch.Tensor):
        self.prototypes = torch.empty((labels.shape[0], self.embedding_size), device=labels.device)
        self.var = torch.empty((labels.shape[0], self.embedding_size), device=labels.device)
        self.alpha_prime = torch.empty((labels.shape[0], self.embedding_size), device=labels.device)
        self.beta_prime = torch.empty((labels.shape[0], self.embedding_size), device=labels.device)
        
    def update_prototype(self, x: torch.Tensor, y:int, embed=True):
        self.shots = x.shape[0]
        if(embed):
            embedded = self.back_bone(x)
        else:
            embedded = x
        proto = embedded.mean(axis=0)
        self.prototypes[y] = proto

        n = x.shape[0]
        beta = self.beta
        alpha = self.alpha
        sq = (embedded - embedded.mean(dim=0, keepdim=True))**2

        self.alpha_prime[y] = alpha + n/2
        self.beta_prime[y] =  beta + 1/2 * sq.sum(dim=0, keepdim=True)

        self.var[y] = (self.beta_prime[y]) / (self.alpha_prime[y] - 1)

    def inv_gamma_nll(self):
        beta = self.beta
        alpha = self.alpha
        z = alpha*beta.log() - torch.lgamma(alpha)
        p = (-alpha-1)*self.var.log() - (beta/self.var)
        return -(z + p).mean()