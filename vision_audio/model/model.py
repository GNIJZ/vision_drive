import torch
from torch import nn
# from torchsummary import summary

class InvertedBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(InvertedBottleneck, self).__init__()
        self.leaky = nn.LeakyReLU(0.2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_channels)
        )
        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d((kernel_size - 1) // 2),
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, padding=0,
                      bias=False),
            nn.BatchNorm2d(in_channels)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        m = self.conv2(self.leaky(self.conv1(x)))
        m = self.leaky(self.conv3(self.leaky(m + x)))
        return m

class EmotionNet(nn.Module):
    def __init__(self, num_emotions):
        super(EmotionNet, self).__init__()
        self.num_emotions = num_emotions
        self.layer1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, 24, 3, 2, bias=False),  # output_size = 24x24
            nn.BatchNorm2d(24),
            nn.LeakyReLU(0.2)
        )
        self.layer2 = InvertedBottleneck(24, 40, 3 )  # output_size = 12x12
        self.layer3 = InvertedBottleneck(40, 56, 3 )  # output_size = 6x6
        self.layer4 = InvertedBottleneck(56, 72, 3)  # output_size = 6x6

        self.layer5 = InvertedBottleneck(72, 88, 3)  # output_size = 6x6
        self.layer6 = InvertedBottleneck(88, 104, 3)  # output_size = 6x6
        self.avgpool = nn.AvgPool2d(4, 1, 0)  # output_size = 1x1
        self.layer7 = nn.Linear(78, self.num_emotions)
        self.pool = nn.MaxPool2d(3, 2, 1)
        self.feature_face = nn.Linear(6,6)
        self.leaky_relu = nn.LeakyReLU(0.2)


    def forward(self, x,pose_feature):

        x = self.layer4(self.pool(self.layer3(self.pool(self.layer2(self.layer1(x))))))
        # x=self.pool(self.layer6(self.pool(self.layer5(x))))
        x=self.pool(x)
        x=self.avgpool(x)
        print(x.shape)
        pose_feature=self.feature_face(pose_feature)
        pose_feature = self.leaky_relu(pose_feature)
        x = x.reshape(-1, 72)

        x = torch.cat((x, pose_feature), dim=1)
        x = self.layer7(x)
        return x

# num_emotions = 2
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = EmotionNet(num_emotions).to(device)
#
# # 创建一个输入样本并移动到相同设备
# dummy_input = torch.randn(512, 1, 64, 64).to(device)
#
# x=model(dummy_input)
# print(x.shape)
# # 打印模型的详细信息
# summary(model, (1, 48, 48))