import torch
import math
import numpy as np
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F


Norm = nn.LayerNorm


class CRM(nn.Module):
    def __init__(self, channel_in=512):

        super(CRM, self).__init__()
        self.query_transform = nn.Conv2d(channel_in, channel_in, kernel_size=1, stride=1, padding=0) 
        self.key_transform = nn.Conv2d(channel_in, channel_in, kernel_size=1, stride=1, padding=0) 

        self.scale = 1.0 / (channel_in ** 0.5)

        self.conv6 = nn.Conv2d(channel_in, channel_in, kernel_size=1, stride=1, padding=0) 

        for layer in [self.query_transform, self.key_transform, self.conv6]:
            weight_init.c2_msra_fill(layer)

    def forward(self, x5):
        # x: B,C,H,W
        B, C, H5, W5 = x5.size()

        x_query = self.query_transform(x5).view(B, C, -1)

        # x_query: B,HW,C
        x_query = torch.transpose(x_query, 1, 2).contiguous().view(-1, C) # BHW, C
        # x_key: B,C,HW
        x_key = self.key_transform(x5).view(B, C, -1)

        x_key = torch.transpose(x_key, 0, 1).contiguous().view(C, -1) # C, BHW

        # W = Q^T K: B,HW,HW
        x_w = torch.matmul(x_query, x_key) 
        x_w = x_w.view(B*H5*W5, B, H5*W5)
        x_w = torch.max(x_w, -1).values # BHW, B
        x_w = x_w.mean(-1)
        x_w = x_w.view(B, -1) * self.scale # B, HW
        x_w = F.softmax(x_w, dim=-1) # B, HW
        x_w = x_w.view(B, H5, W5).unsqueeze(1) # B, 1, H, W
 
        x5 = x5 * x_w
        x5 = self.conv6(x5)

        return x5


class MHA(nn.Module):
    

    def __init__(self, d_model=512, d_k=512, d_v=512, h=8, dropout=.1, channel_in=512):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(MHA, self).__init__()
        self.query_transform = nn.Conv2d(channel_in, channel_in, kernel_size=1, stride=1, padding=0)
        self.key_transform = nn.Conv2d(channel_in, channel_in, kernel_size=1, stride=1, padding=0)
        self.value_transform = nn.Conv2d(channel_in, channel_in, kernel_size=1, stride=1, padding=0)
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout = nn.Dropout(dropout)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, attention_mask=None, attention_weights=None):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        '''
        B, C, H, W = x.size()
        queries = self.query_transform(x).view(B, -1, C)
        keys = self.query_transform(x).view(B, -1, C)
        values = self.query_transform(x).view(B, -1, C)

        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, -1)
        att = self.dropout(att)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out).view(B, C, H, W)  # (b_s, nq, d_model)
        return out
    


    class AP_MP(nn.Module):
        def __init__(self,stride=2):
        super(AP_MP,self).__init__()
        self.sz=stride
        self.gapLayer=nn.AvgPool2d(kernel_size=self.sz,stride=self.sz)
        self.gmpLayer=nn.MaxPool2d(kernel_size=self.sz,stride=self.sz)

    def forward(self,x1,x2):
        B,C,H,W=x1.size()
        apimg=self.gapLayer(x1)
        mpimg=self.gmpLayer(x2)
        byimg=torch.norm(abs(apimg-mpimg),p=2,dim=1,keepdim=True)
        return byimg


class DBEModule(nn.Module):
    def __init__(self,channel):
        super(DBEModule,self).__init__()
        self.channel=channel

        self.conv1=BasicConv2d(channel,channel,3,padding=1)
        self.conv2=BasicConv2d(channel,channel,3,padding=1)

        self.CA1=ChannelAttention(self.channel)
        self.CA2=ChannelAttention(self.channel)
        self.SA1=SpatialAttention()
        self.SA2=SpatialAttention()

        self.glbamp=AP_MP()

        self.conv=BasicConv2d(channel*2+1,channel,kernel_size=1,stride=1)
        
        self.upsample2 = nn.Upsample(scale_factor=2 , mode='bilinear', align_corners=True)
        self.upSA=SpatialAttention()

    def forward(self,x,up):
        x1=self.conv1(x)
        x2=self.conv2(x)
        if(torch.is_tensor(up)):
            x2=x2*self.upSA(self.upsample2(up))+x2
            
        x1=x1+x1*self.CA1(x1)
        x2=x2+x2*self.CA2(x2)

        nx1=x1+x1*self.SA2(x2)
        nx2=x2+x2*self.SA1(x1)

        gamp=self.upsample2(self.glbamp(nx1,nx2))

        res=self.conv(torch.cat([nx1,gamp,nx2],dim=1))

        return res+x

class TransBasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=2, stride=2, padding=0, dilation=1, bias=False):
        super(TransBasicConv2d, self).__init__()
        self.Deconv = nn.ConvTranspose2d(in_planes, out_planes,
                                         kernel_size=kernel_size, stride=stride,
                                         padding=padding, dilation=dilation, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.Deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

def weight_init(module):
    for n, m in module.named_children():
        print('initialize: '+n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, nn.ReLU):
            pass
        else:
            m.initialize()


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)  # input is BCHW, output is BC11

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))  # input is BCHW, output is BC11
        out = max_out
        return self.sigmoid(out)


class LayAtt(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(LayAtt, self).__init__()
        # self.conv0 = nn.Conv2d(1, 1, 7, padding=3, bias=True)
        self.conv1 = nn.Sequential(BasicConv2d(in_channel, in_channel//2, 3, padding=1),
                                   BasicConv2d(in_channel//2, in_channel, 3, padding=1))
        self.channel = out_channel

    def forward(self, x, y):
        # a = torch.sigmoid(self.conv0(y))  # y is B 1 H W
        a = torch.sigmoid(y)  # y is B 1 H W
        x_att = self.conv1(a.expand(-1, self.channel, -1, -1).mul(x))  # -1 means not changing the size of that dimension
        x = x #+ x_att
        return x

    def initialize(self):
        weight_init(self)

class RA(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RA, self).__init__()
        self.convert = nn.Conv2d(in_channel, out_channel, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(True)
        self.convs = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, padding=1, bias=False), nn.BatchNorm2d(out_channel), nn.ReLU(True),
            nn.Conv2d(out_channel, out_channel, 3, padding=1, bias=False), nn.BatchNorm2d(out_channel), nn.ReLU(True),
            nn.Conv2d(out_channel, out_channel, 3, padding=1, bias=False), nn.BatchNorm2d(out_channel), nn.ReLU(True),
            nn.Conv2d(out_channel, 1, 3, padding=1),
        )
        self.channel = out_channel

    def forward(self, x, y):
        a = torch.sigmoid(-y)  # y is B 1 H W
        x = self.relu(self.bn(self.convert(x)))
        x = a.expand(-1, self.channel, -1, -1).mul(x)  # -1 means not changing the size of that dimension
        y = y + self.convs(x)
        return y

    def initialize(self):
        weight_init(self)


class NodeAtt(nn.Module):
    def __init__(self, in_channels):
        super(NodeAtt, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(1 * in_channels, in_channels),
                                 nn.ReLU(),
                                 nn.Linear(in_channels, 1))
        self.lin = nn.Linear(1 * in_channels, in_channels)

    def forward(self, x):  # x has shape [N, 1*in_channels]
        nodeatt = torch.sigmoid(self.mlp(x))  # has shape [N, 1]
        x_out = self.lin(x * nodeatt) + x   # [N, in_channels]
        return x_out

class CB1(nn.Module):
    def __init__(self, in_channels, out_channels, use_bn=True, use_relu=True):
        super(CB1, self).__init__()
        self.use_bn = use_bn
        self.use_relu = use_relu
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        if self.use_bn:
            self.bn = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        y = self.conv(x)
        if self.use_bn:
            y = self.bn(y)
        if self.use_relu:
            y = F.relu(y)
        return y
    
class CB3(nn.Module):
    def __init__(self, in_channels, out_channels, use_bn=True, use_relu=True):
        super(CB3, self).__init__()
        self.use_bn = use_bn
        self.use_relu = use_relu
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        if self.use_bn:
            self.bn = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        y = self.conv(x)
        if self.use_bn:
            y = self.bn(y)
        if self.use_relu:
            y = F.relu(y)
        return y

class CWS(nn.Module):
    def __init__(self, in_channels):
        super(CWS, self).__init__()
        self.channel_reduction = CB1(in_channels*2, in_channels, True, True)
        self.importance_estimator = nn.Sequential(ChannelAttention(in_channels),
                                   CB3(in_channels, in_channels//2, True, True),
                                   CB3(in_channels//2, in_channels//2, True, True),
                                   CB3(in_channels//2, in_channels, True, False), nn.Sigmoid())
    def forward(self, group_semantics, individual_features):
        ftr_concat_reduc = self.channel_reduction(torch.cat((group_semantics, individual_features), dim=1)) 
        P = self.importance_estimator(ftr_concat_reduc) 
        co_saliency_features = group_semantics * P + individual_features * (1-P) 
        return co_saliency_features  

    
class CD(nn.Module):
    def __init__(self, in_channels):
        super(CD, self).__init__()
        self.channel_reduction = CB1(in_channels, in_channels//2, True, True)
        self.deconv = CB3(in_channels//2, in_channels//2, True, True)
    def forward(self, X): 
        [B, M, D, H, W] = X.size()
        X_US2 = self.deconv(US2(self.channel_reduction(X.view(B*M, D, H, W)))) 
        return X_US2.view(B, M, D//2, 2*H, 2*W)
    
    
class CosalHead(nn.Module):
    def __init__(self, in_channels):
        super(CosalHead, self).__init__()
        self.output = nn.Sequential(CB3(in_channels, in_channels*4, True, True),
                           CB3(in_channels*4, in_channels*4, True, True),
                           CB3(in_channels*4, 1, False, False), nn.Sigmoid())
    def forward(self, x):
        return self.output(x)