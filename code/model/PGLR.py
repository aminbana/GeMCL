from model.BaseModel import BaseModel
import torch.nn as nn
import torch
import torch.nn.functional as F


class PGLR(BaseModel):
    def __init__(self, args):
        super(PGLR, self).__init__(args)
        self.finetuning_lr = args.inner_lr
        args.dist = 'norm-dot'
        
    def reset_episode(self, labels:torch.Tensor):
        self.prototypes = torch.ones((labels.shape[0], self.embedding_size), device=labels.device, requires_grad = True)
    
    def update_prototype(self, x: torch.Tensor, y:int, embed=True):
        self.shots = x.shape[0]
        if(embed):
            embedded = self.back_bone(x)
        else:
            embedded = x
        with torch.enable_grad():
            self.inner_loop(embedded,  y)

    def inner_loop(self, embedded, y):
        device = next(self.parameters()).device
        target = torch.tensor([y]).to(device)
        grad_mask = torch.zeros(self.prototypes.shape[0], 1).to(device)
        grad_mask[y] = 1

        for i in range(embedded.shape[0]):
            if(not self.training):
                self.prototypes.requires_grad = True
            scores = self.classify(embedded[i:i+1])
            
            loss = nn.CrossEntropyLoss()(scores, target)

            grads = torch.autograd.grad(loss, self.prototypes)[0] * grad_mask

            if(not self.training):
                self.prototypes = self.prototypes.detach() - grads.detach()*self.finetuning_lr
            else:
                self.prototypes = self.prototypes - grads*self.finetuning_lr
