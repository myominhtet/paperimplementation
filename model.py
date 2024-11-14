import torch
import torch.nn as nn

class wafernet(nn.Module):
    def __init__(self, in_channel, num_classes=38, drop=0.5):
        super(wafernet, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=8, kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.Dropout2d(drop), 
            nn.Conv2d(in_channels=8, out_channels=8,kernel_size=(3,3), stride=1, padding=1, groups=8),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=1),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.Dropout2d(drop), 
            nn.Conv2d(in_channels=16, out_channels=16,kernel_size=(3,3), stride=1, padding=1, groups=16),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.Dropout2d(drop), 
            nn.Conv2d(in_channels=16, out_channels=16,kernel_size=(3,3), stride=1, padding=1, groups=16),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.Dropout2d(drop), 
            nn.Conv2d(in_channels=32, out_channels=32,kernel_size=(3,3), stride=1, padding=1, groups=8),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        self.block5 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.Dropout2d(drop), 
            nn.Conv2d(in_channels=64, out_channels=64,kernel_size=(3,3), stride=1, padding=1, groups=64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1),
            nn.BatchNorm2d(64),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64,128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3), 
            nn.Linear(128, num_classes) 
        )
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.classifier(x)
        return(x)
model = wafernet(in_channel=1,num_classes=38, drop=0.5)
# input_tensor = torch.randn(1, 3, 64, 64)
# output = model(input_tensor)
# print(output.shape)