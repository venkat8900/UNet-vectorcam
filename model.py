import torch
import torch.nn as nn
import torchvision.transforms.functional as transforms


class DoubleConv(nn.Module):
    """
    params: in_channels
    params: out_channels
    
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
        )
    
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    """
    params: in_channels: 3 by default as we input image with three channels
            out_channels: for binary segmentation: out_channels = 1, for multi class, out_channels = num_classes 
            channels: number of channels in each layer of network
    """
    def __init__(self, in_channels = 3, out_channels = 1, channels = [64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.up = nn.ModuleList()
        self.down = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)

        # encoder blocks
        for c in channels:
            self.down.append(DoubleConv(in_channels, c))
            in_channels = c
        
        # decoder blocks
        for c in channels[::-1]:
            self.up.append(nn.ConvTranspose2d(c * 2, c, kernel_size = 2, stride = 2))
            self.up.append(DoubleConv(c * 2, c))

        self.bottleneck = DoubleConv(channels[-1], channels[-1] * 2)
        self.final = nn.Conv2d(channels[0], out_channels, kernel_size = 1)
    
    def forward(self, x):
        skips = []

        for d in self.down:
            x = d(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        skips.reverse()

        for idx in range(0, len(self.up), 2):
            x = self.up[idx](x)
            skip_con = skips[idx//2]

            # if the shape doesn't match during up-sampling
            if x.shape != skip_con.shape:
                x = transforms.resize(x, size = skip_con.shape[2:])

            concat_skip = torch.cat((skip_con, x), dim = 1)
            x = self.up[idx + 1](concat_skip)
        
        return self.final(x)


def test():
    x = torch.randn((3, 1, 160, 160))
    model = UNet(in_channels = 1, out_channels = 1)
    preds = model(x)
    assert preds.shape == x.shape

if __name__ == '__main__':
    test()
            
