import torch

from models.Attention import *
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self,num_hiddens,dropout,max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout=nn.Dropout(dropout)
        self.P=torch.zeros((1,max_len,num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)
    def forward(self,X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)

class MultiHeadAttention(nn.Module):
    def __init__(self,key_size,value_size,query_size,num_heads,num_hiddens,dropout,bias=False):
        #后续的调试中可以让隐藏层大小不一样
        super(MultiHeadAttention, self).__init__()
        self.num_heads=num_heads
        self.attention=DotProductAttention(dropout)
        self.w_q=nn.Linear(query_size,num_hiddens,bias=bias)
        self.w_k=nn.Linear(key_size,num_hiddens,bias=bias)
        self.w_v=nn.Linear(value_size,num_hiddens,bias=bias)
        # self.w_o=nn.Linear(num_hiddens,num_hiddens,bias=bias)
        self.w_o = nn.Linear(num_hiddens, query_size, bias=bias)
    def forward(self,queries,keys,values,valid_lens):
        queries=transpose_qkv(self.w_q(queries),self.num_heads)
        keys=transpose_qkv(self.w_k(keys),self.num_heads)
        values=transpose_qkv(self.w_v(values),self.num_heads)
        if valid_lens is not None:
            valid_lens=torch.repeat_interleave(valid_lens,repeats=self.num_heads,dim=0)
        output=self.attention(queries,keys,values,valid_lens)
        output_concat=transpose_output(output,self.num_heads)
        return self.w_o(output_concat)

class AddNorm(nn.Module):
    '''加法和层归一化层'''
    def __init__(self,normalized_shape,dropout):
        super(AddNorm, self).__init__()
        self.dropout=nn.Dropout(dropout)
        self.layernorm=nn.LayerNorm(normalized_shape)
    def forward(self,x,y):
        return self.layernorm(self.dropout(y)+x)

class AFF(nn.Module):
    '''
    多特征融合 AFF
    '''

    def __init__(self, channels=64, r=4):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv1d(channels, inter_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(inter_channels, channels, kernel_size=1),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, inter_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(inter_channels, channels, kernel_size=1),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        xo = 2 * x * wei + 2 * residual * (1 - wei)
        return xo

class AddNorm2(nn.Module):
    '''加法和层归一化层'''
    def __init__(self,normalized_shape,dropout):
        super(AddNorm2, self).__init__()
        self.dropout=nn.Dropout(dropout)
        self.aff=AFF(128)
        self.layernorm=nn.LayerNorm(normalized_shape)
    def forward(self,x,y):
        return self.layernorm(self.dropout(self.aff(x.permute(0,2,1),y.permute(0,2,1)).permute(0,2,1)))


class PositionWiseFFN(nn.Module):
    '''逐位前馈网络'''
    def __init__(self,input_size,output_size,num_hiddens):
        super(PositionWiseFFN, self).__init__()
        self.dense1=nn.Linear(input_size,num_hiddens)
        self.relu=nn.ReLU()
        self.dense2=nn.Linear(num_hiddens,output_size)
    def forward(self,x):
        return self.dense2(self.relu(self.dense1(x)))

class InternalExternalAttention(nn.Module):
    def __init__(self,num_hiddens,law_nums):
        super(InternalExternalAttention, self).__init__()
        self.w=nn.Linear(num_hiddens,law_nums,bias=True)
        self.law_nums=law_nums

    def InternalExternalSoftmax(self,X, valid_lens):
        j=0
        for i in X:
            X[j].copy_ = mask_softmax(i, valid_lens)
            j =j+1
        return X

    def forward(self, X, Laws, X_valid_lens, Law_valid_lens):
        X=X.unsqueeze(1)

        a=self.InternalExternalSoftmax(torch.matmul(Laws, X.permute(0, 1, 3, 2)).permute(1, 0, 2, 3), X_valid_lens).permute(1, 0, 2, 3)
        b=self.InternalExternalSoftmax(torch.matmul(X, Laws.permute(0, 2, 1)), Law_valid_lens)
        D=torch.matmul(b,Laws)

        feature=torch.cat((torch.repeat_interleave(X, self.law_nums, 1), D), 3)
        feature=F.avg_pool2d(torch.matmul(a, feature),(Laws.shape[1],1)).squeeze(2)
        feature=F.avg_pool1d(feature,2)

        X = X.squeeze(1)
        t=self.w(X)
        y=F.softmax(t,dim=-1)

        return torch.bmm(y,feature)+X


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        # 利用1x1卷积代替全连接
        self.fc1   = nn.Conv1d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv1d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv1d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class Cbam_block(nn.Module):
    #ratio要小于channel的值
    def __init__(self, channel, ratio=8, kernel_size=7):
        super(Cbam_block, self).__init__()
        # 用于让模型认识每个字,所有字的通一维特征的采用相同的权重
        self.channelattention = ChannelAttention(channel, ratio=ratio)
        #关注整句话，输出的权重是（batch，1，step），同一个字的特征维度采用相同的权重，这样不同的字就会突出不同的重要度，不重要的词的权重会分配很少，这样就在后续的运算中越来越小
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        #x(batch,hidden,steps),channelattention(x):(batch,hiddens,1)
        self.atttention_channel=self.channelattention(x)
        x = x * self.channelattention(x)
        self.atttention_spatial = self.spatialattention(x)
        x = x * self.spatialattention(x)
        return x

class BiRNN(nn.Module):
    def __init__(self,embed_size,num_hiddens,num_layers,**kwargs):
        super(BiRNN, self).__init__()
        self.encoder=nn.GRU(embed_size,num_hiddens,num_layers=num_layers,bidirectional=True)
        self.linear=nn.Linear(2*num_hiddens,embed_size)
    def forward(self,inputs):
        self.encoder.flatten_parameters()
        outputs,_=self.encoder(inputs.permute(1,0,2))
        return self.linear(outputs.permute(1,0,2))

def sequence_mask(x,valid_len,value=0):
    '''屏蔽pad项'''
    maxlen=x.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=x.device)[None, :] < valid_len[:, None]
    x = torch.where(~mask, value, x)
    # x[~mask]=value
    return x


def mask_softmax(x,valid_lens):
    '''屏蔽pad，将其softmax后的值置为0'''
    if valid_lens is None:
        return nn.functional.softmax(x,dim=-1)
    else:
        shape=x.shape
        if valid_lens.dim()==1:
            valid_lens=torch.repeat_interleave(valid_lens,shape[1])
        else:
            valid_lens=valid_lens.reshape(-1)
    x=sequence_mask(x.reshape(-1,shape[-1]),valid_lens,value=-1e6)
    return nn.functional.softmax(x.reshape(shape),dim=-1)

if __name__ == '__main__':
    # test=InternalExternalAttention(5,5)
    # a=torch.ones((6,5,5))
    # lens=torch.tensor([1,2,3,4,5,5])
    # b=torch.ones(5,7,5)
    # lens2=torch.tensor([1,2,3,4,5])
    # test(a,b,lens,lens2)
    a=Cbam_block(128)
    b=torch.ones((2,128,10))
    print(a(b).shape)
    # a=torch.ones(6,1,50)
    # b=torch.tensor([1,2,3,4,5,6])
    # c=mask_softmax(a,b)
    # print(c)
    # print(c.shape)