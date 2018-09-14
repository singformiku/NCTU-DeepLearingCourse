'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''


import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module): #繼承nn.Module
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__() #使用nn.Module的基底類別
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
 

#class Bottleneck(nn.Module):
#    expansion = 4
#
#    def __init__(self, in_planes, planes, stride=1):
#        super(Bottleneck, self).__init__()
#        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
#        self.bn1 = nn.BatchNorm2d(planes)
#        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#        self.bn2 = nn.BatchNorm2d(planes)
#        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
#        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
#
#        self.shortcut = nn.Sequential()
#        if stride != 1 or in_planes != self.expansion*planes:
#            self.shortcut = nn.Sequential(
#                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
#                nn.BatchNorm2d(self.expansion*planes)
#            )
#
#    def forward(self, x):
#        out = F.relu(self.bn1(self.conv1(x)))   #x   = in_planes   out = planes
#        out = F.relu(self.bn2(self.conv2(out))) #out = in_planes   out = planes
#        out = self.bn3(self.conv3(out))
#        out += self.shortcut(x)                 #out = out + self.shortcut(x)
#        out = F.relu(out)
#        return out

#     .Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
#           (in_channels=3,out_channels=4,kernel_size=3,groups=1)
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 25#64

        self.conv1 = nn.Conv2d(3, 25, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(25)
        self.layer1 = self._make_layer(block, 25, num_blocks[0], stride=1) #64
        self.layer2 = self._make_layer(block, 50, num_blocks[1], stride=2)#128
        self.layer3 = self._make_layer(block, 100, num_blocks[2], stride=2)#256
		#############################################################################
        #self.layer4 = self._make_layer(block, 160, num_blocks[3], stride=2)#512
		#############################################################################
        self.linear = nn.Linear(400*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
		#############################################################################
        #out = self.layer4(out)
		#############################################################################
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
 

def ResNet18():#20
    return ResNet(BasicBlock, [3,3,3])# 2 2 2 2

#def ResNet34():
#    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():#56
    return ResNet(BasicBlock, [9,9,9])#3 4 6 3

def ResNet101():#110
    return ResNet(BasicBlock, [18,18,18])#3 4 23 6

#def ResNet152():
#    return ResNet(Bottleneck, [3,8,36,3])


def test():
    net = ResNet18()
    y = net(torch.randn(1,3,32,32))
    print(y.size())

# test()