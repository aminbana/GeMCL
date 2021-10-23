import torch.nn as nn
import torch
import torch.nn.functional as F

class OMLNet(nn.Module):
    def __init__(self, input_channels=1, channels = 256):
        super(OMLNet, self).__init__()
        

        self.layers = nn.Sequential(
            torch.nn.Conv2d(in_channels = input_channels, out_channels = channels, kernel_size = 3, stride = 2, padding = 0),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels = channels, out_channels = channels, kernel_size = 3, stride = 1, padding = 0),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels = channels, out_channels = channels, kernel_size = 3, stride = 2, padding = 0),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels = channels, out_channels = channels, kernel_size = 3, stride = 1, padding = 0),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels = channels, out_channels = channels, kernel_size = 3, stride = 2, padding = 0),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels = channels, out_channels = channels, kernel_size = 3, stride = 2, padding = 0),
            torch.nn.ReLU(),
        )


        with torch.no_grad():
            for m in self.layers:
                if isinstance(m,nn.Conv2d):
                    # print (m.__)
                    torch.nn.init.kaiming_normal_(m.weight)
                    torch.nn.init.zeros_(m.bias)

    def get_embedding_size(self, input_size = (1,84,84)):
        
        device = next(self.parameters()).device
        assert input_size[2] == 84
        x = torch.rand([2,*input_size]).to(device)
        # print (x.shape)
        with torch.no_grad():
            output = self.forward(x)
            emb_size = output.shape[-1]
        
        del x,output
        torch.cuda.empty_cache()

        return emb_size

    def forward(self, x):
        return self.layers (x).reshape ([x.shape[0] , -1])

if __name__=="__main__":
    model = OMLNet (256).cuda()
    x = torch.rand([2,1,84,84]).cuda()
    # print ()
    assert model(x).shape[-1] == model.get_embedding_size()