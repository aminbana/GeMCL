import torch.nn as nn
import torch

def conv_block(in_channels, out_channels):
    '''
    returns a block conv-bn-relu-pool
    '''
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )


class ProtoNetBack(nn.Module):
    def __init__(self, input_channels = 1):
        super(ProtoNetBack, self).__init__()
        self.layers = nn.Sequential(
            conv_block(input_channels, 64),
            conv_block(64, 64),
            conv_block(64, 64),
            conv_block(64, 64),
        )

    def get_embedding_size(self, input_size = (1,28,28)):
        device = next(self.parameters()).device
        x = torch.rand([2,*input_size]).to(device)
        with torch.no_grad():
            output = self.forward(x)
            emb_size = output.shape[-1]
        
        del x,output
        torch.cuda.empty_cache()

        return emb_size

    def forward(self, x):
        return self.layers (x).reshape ([x.shape[0] , -1])
    
    