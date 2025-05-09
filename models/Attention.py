import torch
from torch import  nn
import math
from utils import *
class DotProductAttention(nn.Module):
    """缩放点积注意力"""
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
    # queries的形状：(batch_size，查询的个数，d)
    # keys的形状：(batch_size，“键－值”对的个数，d)
    # values的形状：(batch_size，“键－值”对的个数，值的维度)
    # valid_lens的形状:(batch_size，)或者(batch_size，查询的个数)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        # 设置transpose_b=True为了交换keys的最后两个维度
        scores = torch.bmm(queries, keys.transpose(1,2)) / math.sqrt(d)
        self.attention_weights = mask_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)

class AdditiveAttention(nn.Module):
    """加性注意力"""
    def __init__(self,key_size,query_size,num_hiddens,dropout):
        super(AdditiveAttention,self).__init__()
        self.w_k=nn.Linear(key_size,num_hiddens,bias=False)
        self.w_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)
        self.tanh = nn.Tanh()
        self.dropout=nn.Dropout(dropout)
    def forward(self,queries, keys, values, valid_lens=None):
        queries,keys=self.w_q(queries),self.w_k(keys)
        features=queries.unsqueeze(2)+keys.unsqueeze(1)
        features=self.tanh(features)
        scores=self.w_v(features).squeeze(-1)
        self.attention_weights=mask_softmax(scores,valid_lens)
        return torch.bmm(self.dropout(self.attention_weights),values)











