from typing import List, Callable

import numpy as np
import matplotlib.pyplot as plt
from torchsummary import summary
import torch.nn.functional as F
import torch
from torch import Tensor
import torch.nn as nn








import torch
import torch.nn.functional as F

class ImprovedELU(torch.nn.Module):
    """
    Applies the Exponential Linear Unit (ELU) function, element-wise, with improved smoothness.

    Improved ELU is defined as:

        ImprovedELU(x) = max(0, x) + min(0, alpha * (exp(x) - 1))

    Args:
        alpha: the alpha value for the ELU formulation. Default: 1.0

    Shape:
        - Input: (*), where * means any number of dimensions.
        - Output: (*), same shape as the input.
    """
    def __init__(self, alpha=1.0):
        super(ImprovedELU, self).__init__()
        self.alpha = alpha

    def forward(self, input):
        return F.relu(input) + F.elu(-F.relu(-input), self.alpha)

    def extra_repr(self):
        return 'alpha={}'.format(self.alpha)









class ACBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ACBlock, self).__init__()
        self.conv1x3 = nn.Conv2d(in_channels, out_channels, (1, 3), 1, (0, 1))
        self.conv3x1 = nn.Conv2d(in_channels, out_channels, (3, 1), 1, (1, 0))
        # self.conv3x3 = nn.Conv2d(in_channels, out_channels, (3, 3), 1, (1, 1))

    def forward(self, x):
        x = self.conv3x1(x)
        # print(x)
        x = self.conv1x3(x)
        # print(x)
        # conv3x3 = self.conv3x3(x)
        return x


class SimAM(torch.nn.Module):
    def __init__(self, channels = None, e_lambda = 1e-4):
        super(SimAM, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):

        b, c, h, w = x.size()
        
        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2,3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2,3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)



class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAttention(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        

    def forward(self, x):
        identity = x
        
        n,c,h,w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out



def channel_shuffle(x: Tensor, groups: int) -> Tensor:

    batch_size, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    # reshape
    # [batch_size, num_channels, height, width] -> [batch_size, groups, channels_per_group, height, width]
    x = x.view(batch_size, groups, channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batch_size, -1, height, width)

    return x


class SeAttention(nn.Module):
    def __init__(self, channel_num, r=4):
        super(SeAttention, self).__init__()
        self.channel_num = channel_num
        self.r = r
        self.inter_channel = int( float(self.channel_num) / self.r)
        self.fc_e1 = torch.nn.Linear(channel_num, self.inter_channel)
        #self.bn_e1 = torch.nn.BatchNorm2d(self.inter_channel)
        self.relu_e1 = nn.ReLU(inplace=True)
        self.fc_e2 = torch.nn.Linear(self.inter_channel, channel_num)
        #self.bn_e2 = torch.nn.BatchNorm2d(channel_num)

    def forward(self, x):
        y = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze()
        y = self.fc_e1(y)
        #y = self.bn_e1(y)
        y = self.relu_e1(y)
        y = self.fc_e2(y)
        #y = self.bn_e2(y)
        y = torch.sigmoid(y).unsqueeze(-1).unsqueeze(-1)
        #y = y.unsqueeze(-1)
        # print("此处增加注意力机制")
        return x*y



class InvertedResidual(nn.Module):
    def __init__(self, input_c: int, output_c: int, stride: int):
        super(InvertedResidual, self).__init__()

        if stride not in [1, 2]:
            raise ValueError("illegal stride value.")
        self.stride = stride

        assert output_c % 2 == 0
        branch_features = output_c // 2
        # 当stride为1时，input_channel应该是branch_features的两倍
        # python中 '<<' 是位运算，可理解为计算×2的快速方法
        assert (self.stride != 1) or (input_c == branch_features << 1)

        if self.stride == 2:
            self.branch1 = nn.Sequential(
                # nn.AvgPool2d(kernel_size=5,stride=2,padding=2), #修改
                self.depthwise_conv(input_c, input_c, kernel_s=5, stride=self.stride, padding=2),#修改kernel3变成5，修改padding1变成2
                nn.BatchNorm2d(input_c),
                nn.Conv2d(input_c, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True)
                
            )
            # self.branch12=(
            #     #添加
            #     ACBlock(branch_features,branch_features),
            #     nn.ReLU(inplace=True),
            #     nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=2, bias=False),
            #     nn.BatchNorm2d(branch_features),
            #     nn.ReLU(inplace=True)
            # )
            
            
        else:
            self.branch1 = nn.Sequential(       
               
            )
            # self.branch12=(
            #     #添加
            #     nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=2, bias=False),
            #     nn.BatchNorm2d(branch_features),
            #     nn.ReLU(inplace=True),
            #     ACBlock(branch_features,branch_features),
            #     nn.BatchNorm2d(branch_features),
            #     nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=2, bias=False),
            #     nn.BatchNorm2d(branch_features),
            #     nn.ReLU(inplace=True)
            # )
            # conv3x1=nn.Conv2d(kernel_size=(3,1), stride=1, padding=2, bias=False),
            # conv1x3=nn.Conv2d(kernel_size=(1,3), stride=1, padding=2, bias=False),
            # self.branch1_1=nn.Sequential(
            #     nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=2, bias=False),
            #     nn.BatchNorm2d(branch_features),
            #     nn.ELU(inplace=True),
            #     # x1=conv3x1(branch_features),
            #     # x2=conv1x3(branch_features),
            #     # branch_features=torch.add(x1,x2),
            #     # nn.BatchNorm2d(branch_features),
            #     # nn.BatchNorm2d(branch_features),
            #     # nn.ELU(inplace=True)
            #     # conv3x1 = nn.Conv2d(32,32,(3,1))    # 3x1卷积
            #     # conv1x3 = nn.Conv2d(32,32,(1,3))    # 1x3卷积
            # )
        self.branch2 = nn.Sequential(
            nn.Conv2d(input_c if self.stride > 1 else branch_features, branch_features, kernel_size=1,
                    stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            self.depthwise_conv(branch_features, branch_features, kernel_s=5, stride=self.stride, padding=2),#修改kernel3变成5，修改padding1变成2
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True)
        
            
        )
       

    @staticmethod
    def depthwise_conv(input_c: int,
                       output_c: int,
                       kernel_s: int,
                       stride: int = 1,
                       padding: int = 0,
                       bias: bool = False) -> nn.Conv2d:
        return nn.Conv2d(in_channels=input_c, out_channels=output_c, kernel_size=kernel_s,
                         stride=stride, padding=padding, bias=bias, groups=input_c)       #dw卷积与普通卷积的不同在于

    def forward(self, x: Tensor) -> Tensor:
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
            
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

        return out



class NMF(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(NMF, self).__init__()
        self.W = nn.Parameter(torch.randn(input_channels, output_channels))
        self.H = nn.Parameter(torch.randn(output_channels, 1))
    
    def forward(self, X):
        # X: (batch_size, input_channels, height, width)
        
        # Reshape input tensor for NMF
        X_reshape = X.view(X.size(0), X.size(1), -1)  # (batch_size, input_channels, height*width)
        
        # Non-negative matrix factorization
        X_hat = torch.matmul(self.W, self.H)  # (input_channels, 1)
        
        # Reshape output tensor
        output = X_hat.view(X.size(0), -1, X.size(2), X.size(3))  # (batch_size, output_channels, height, width)
        
        return output

class ShuffleNetV2(nn.Module):
    def __init__(self,
                 stages_repeats: List[int],
                 stages_out_channels: List[int],
                 num_classes: int = 1000,
                 inverted_residual: Callable[..., nn.Module] = InvertedResidual):
        super(ShuffleNetV2, self).__init__()

        if len(stages_repeats) != 3:
            raise ValueError("expected stages_repeats as list of 3 positive ints")
        if len(stages_out_channels) != 5:
            raise ValueError("expected stages_out_channels as list of 5 positive ints")
        self._stage_out_channels = stages_out_channels

        # input RGB image
        input_channels = 3
        output_channels = self._stage_out_channels[0]

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
           
        )
        input_channels = output_channels

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Static annotations for mypy
        self.stage2: nn.Sequential
        self.stage3: nn.Sequential
        self.stage4: nn.Sequential

        stage_names = ["stage{}".format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(stage_names, stages_repeats,
                                                  self._stage_out_channels[1:]):
            seq = [inverted_residual(input_channels, output_channels, 2)]
            for i in range(repeats - 1):
                seq.append(inverted_residual(output_channels, output_channels, 1))
            # seq.append(SeAttention(output_channels))  #修改增加了注意力机制
                # print(stage_names[1])
                if stage_names[1]=="stage3":
                    seq.append(SimAM(output_channels,output_channels))  
                    print("此处增加了注意力机制")
            # seq.append(SimAM(output_channels,output_channels))
                
            
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels

        output_channels = self._stage_out_channels[-1]
        self.conv5 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
            
        )

        self.fc = nn.Linear(output_channels, num_classes)

        #添加
        self.L1 = nn.Sequential(
            nn.Conv2d(24,116,kernel_size=2,stride=2,bias=False),
            torch.nn.BatchNorm2d(116),
            torch.nn.ELU(inplace=True),
            ImprovedELU(),
            )
        self.L2 = nn.Sequential(
            nn.Conv2d(116,232,kernel_size=2,stride=2,bias=False),
            torch.nn.BatchNorm2d(232),
            torch.nn.ELU(inplace=True),
            )
        self.L3 = nn.Sequential(
            nn.Conv2d(232,464,kernel_size=2,stride=2,bias=False),
            torch.nn.BatchNorm2d(464),
            torch.nn.ELU(inplace=True),
            )

        self.softmax = nn.Softmax()
    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x1 = self.conv1(x)      # 32,24,112,112
        x2 = self.maxpool(x1)   # 32,24,56,56
        n = self.L1(x2)         # 32，116，28，28
        x3 = self.stage2(x2)    # 32，116，28，28
        m=n+x3                  # 32，116，28，28 
        
        x4 = self.stage3(m)    # 32，232，14，14
        n1=self.L2(m)
        m1=n1+x4                  # 32，232，14，14
       
        x5 = self.stage4(m1)    # 32,464,7,7
        n2=self.L3(m1)
        
        m2=n2+x5
        x6 = self.conv5(m2)     # 32,1024,7,7
        x7 = x6.mean([2, 3])  # global pool  2，3是高度和宽度两个维度，mean之后高度和宽度就没有了   #32,1024
        x8 = self.fc(x7)      #32,16   16分类 
        
        # m = self.L1(x3) #32,464,14,14
        # n = self.L2(x4) #32,464,14,14
        # print(m.shape)
        # print(n.shape)
        
        
        
        # return x



#下边用来显示特征图
        # x1 = self.conv1(x)
        # x2 = self.maxpool(x1)
        # x3 = self.stage2(x2)
        # x4 = self.stage3(x3)
        # x5 = self.stage4(x4)
        # x6 = self.conv5(x5)
        # x7 = x6.mean([2, 3])  # global pool
        # x8 = self.fc(x7)
        

        # # #添加
        # outputs = []
        # outputs.append(n)
        # # outputs.append(self.aspp(low_level_features))
        # # outputs.append(self.attention(x))
        # # outputs.append(x)

        # for feature_map in outputs:
        #     # 通过一个迭代器来遍历每个特征图
        #     # [N, C, H, W] -> [C, H, W]
        #     im = np.squeeze(feature_map.detach().numpy())#把tensor变成numpy
        #     # [C, H, W] -> [H, W, C]
        #     im = np.transpose(im, [1, 2, 0])
        #     # 对图像的通道进行处理
        #     # show top 12 feature maps
        #     plt.figure()
        
        #     for i in range(64):
        #         ax = plt.subplot(8, 8, i+1)
        #         # [H, W, C]
        #         plt.imshow(im[:, :, i])
        #         plt.imshow(im[:, :, i])
        #         plt.axis('off')
            
        #     plt.show()

        # 添加
        return x8



    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def shufflenet_v2_x0_5(num_classes=1000):
    """
    Constructs a ShuffleNetV2 with 0.5x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`.
    weight: https://download.pytorch.org/models/shufflenetv2_x0.5-f707e7126e.pth

    :param num_classes:
    :return:
    """
    model = ShuffleNetV2(stages_repeats=[4, 8, 4],
                         stages_out_channels=[24, 48, 96, 192, 1024],
                         num_classes=num_classes)

    return model


def shufflenet_v2_x1_0(pretrained=False, num_classes=16):
    """
    Constructs a ShuffleNetV2 with 1.0x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`.
    weight: https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth

    :param num_classes:
    :return:
    """
    model = ShuffleNetV2(stages_repeats=[4, 8, 4],
                         stages_out_channels=[24, 116, 232, 464, 1024],
                         num_classes=num_classes)
    # print(model)
    # if num_classes!=1000:
        
    #     model.classifier = nn.Sequential(  
    #         nn.Dropout(),
    #         nn.Linear(8192,num_classes)
    #     )

    return model
# model=shufflenet_v2_x1_0()
# print(model)

def shufflenet_v2_x1_5(num_classes=1000):
    """
    Constructs a ShuffleNetV2 with 1.0x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`.
    weight: https://download.pytorch.org/models/shufflenetv2_x1_5-3c479a10.pth

    :param num_classes:
    :return:
    """
    model = ShuffleNetV2(stages_repeats=[4, 8, 4],
                         stages_out_channels=[24, 176, 352, 704, 1024],
                         num_classes=num_classes)

    return model


def shufflenet_v2_x2_0(num_classes=1000):
    """
    Constructs a ShuffleNetV2 with 1.0x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`.
    weight: https://download.pytorch.org/models/shufflenetv2_x2_0-8be3c8ee.pth

    :param num_classes:
    :return:
    """
    model = ShuffleNetV2(stages_repeats=[4, 8, 4],
                         stages_out_channels=[24, 244, 488, 976, 2048],
                         num_classes=num_classes)

    return model


