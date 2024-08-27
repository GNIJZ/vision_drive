import torch
from torch import nn
import torch.nn.functional as F
class BasicBlock(nn.Module):
    def __init__(self,in_channels,out_channels,stride):
        super(BasicBlock, self).__init__()
        self.block=nn.Sequential(
            nn.Conv2d(in_channels,out_channels//4,kernel_size=1,stride=stride,padding=0,bias=False),
            nn.BatchNorm2d(out_channels//4),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels//4,out_channels//4,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(out_channels //4),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels //4, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    def forward(self,x):
        out=self.block(x)
        res=self.shortcut(x)
        out += res
        out=F.relu(out)
        return out


class ResNet50(nn.Module):
    def __init__(self,resblock,num_classes):
        super(ResNet50,self).__init__()
        self.inchannels=128
        self.conv1=nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=128,kernel_size=3, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.layer1=self.make_layer(resblock,channels=128,num_blocks=2,stride=1)
        self.layer2=self.make_layer(resblock,channels=256,num_blocks=3,stride=2)
        self.layer3 = self.make_layer(resblock, channels=512,num_blocks=2,stride=2)
        self.layer4 = self.make_layer(resblock, channels=1024,num_blocks=1, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc=nn.Linear(1024,num_classes)



    def make_layer(self,resblock,channels,num_blocks,stride):
        strides=[stride]+[1]*(num_blocks-1)
        layers=[]

        for s in strides:
            layers.append(resblock(self.inchannels,channels,s))
            self.inchannels = channels
        return nn.Sequential(*layers)

    def forward(self,x):                    #[3,96,96]
        out=self.conv1(x)                   #[64,25,25]
        out=self.layer1(out)                #[64,25,25]
        out=self.layer2(out)                #[128,13,13]
        out=self.layer3(out)                #[256,7,7]
        out=self.layer4(out)

        out=F.avg_pool2d(out,4)   #[512,1,1]
        out = out.view(-1,out.size(1)) #转onxx
        out=self.fc(out)
        return out








