import math

import torch
from torch import nn
from Layer import *
import time


class EncoderDecoder(nn.Module):
    def __init__(self,encoder,decoder,**kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder=encoder
        self.decoder=decoder
    def forward(self,enc_x,dec_x,*args):
        enc_output=self.encoder(enc_x,*args)
        dec_state=self.decoder.init_state(enc_output,*args)
        return self.decoder(dec_x,dec_state)

class Encoder(nn.Module):
    def __init__(self,encoder,num_steps,num_hiddens,output,**kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.encoder=encoder
        self.flatten=nn.Flatten()
        self.output=nn.Linear(num_hiddens*num_steps,output)
    def forward(self,enc_x,*args):
        enc_output=self.encoder(enc_x,*args)

        return  self.output(self.flatten(enc_output))


class EncoderCnn2(nn.Module):
    def __init__(self, encoder, embed_size, kernel_size, num_channels, **kwargs):
        super(EncoderCnn2, self).__init__(**kwargs)
        self.encoder = encoder
        self.gru1 = nn.GRU(embed_size, 256, num_layers=4, bidirectional=True)
        self.gru2 = nn.GRU(embed_size, 256, num_layers=4, bidirectional=True)
        self.attention = DotProductAttention(0.3)
        self.output = TextCnn(256 * 4, kernel_size, num_channels)


    def forward(self, enc_x, *args):
        self.gru1.flatten_parameters()
        self.gru2.flatten_parameters()
        enc_output = self.encoder(enc_x, *args)
        outputs1, _ = self.gru1(enc_output.permute(1, 0, 2))
        outputs2, _ = self.gru2(self.encoder.embedding(enc_x).permute(1, 0, 2))
        outputs1 = outputs1.permute(1, 0, 2)
        outputs2 = outputs2.permute(1, 0, 2)
        outputs2 = self.attention(outputs2, outputs2, outputs2, *args)
        encoding1 = torch.cat((outputs1, outputs2), dim=2)
        return self.output(encoding1)


class EncoderBlock(nn.Module):
    def __init__(self,key_size,query_size,value_size,num_hiddens,num_heads,
                 norm_shape,ffn_num_input,ffn_num_hiddens,ffn_num_output,dropout=0,use_bias=False):
        super(EncoderBlock, self).__init__()
        self.attention=MultiHeadAttention(key_size,value_size,query_size,num_heads,num_hiddens,dropout,use_bias)
        self.addnorm1=AddNorm(norm_shape,dropout)
        self.ffn=PositionWiseFFN(ffn_num_input,ffn_num_output,ffn_num_hiddens)
        self.addnorm2=AddNorm(norm_shape,dropout)
    def forward(self,x,valid_lens):
        y=self.addnorm1(x,self.attention(x,x,x,valid_lens))
        return self.addnorm2(y,self.ffn(y))


class EncoderBlockCnn(nn.Module):
    def __init__(self,key_size,query_size,value_size,num_hiddens,num_heads,
                 norm_shape,ffn_num_input,ffn_num_hiddens,ffn_num_output,dropout=0,use_bias=False):
        super(EncoderBlockCnn, self).__init__()
        self.attention=MultiHeadAttention(key_size,value_size,query_size,num_heads,num_hiddens,dropout,use_bias)
        self.attentionCnn=Cbam_block(query_size,8,7)
        self.linear=nn.Linear(2*key_size,key_size)
        self.relu=nn.ReLU()
        self.addnorm1=AddNorm(norm_shape,dropout)
        self.ffn=PositionWiseFFN(ffn_num_input,ffn_num_output,ffn_num_hiddens)
        self.addnorm2=AddNorm(norm_shape,dropout)
    def forward(self,x,valid_lens):
        y1=self.attention(x, x, x, valid_lens)
        y2=self.attentionCnn(x.permute(0,2,1)).permute(0,2,1)
        y=self.relu(self.linear(torch.cat([y1,y2],dim=-1)))
        y=self.addnorm1(x,y)
        return self.addnorm2(y,self.ffn(y))



class TransformerEncoderCnn(nn.Module):
    def __init__(self,vocab_size,key_size,query_size,value_size,embedding_hiddens,num_hiddens,num_heads,
                 norm_shape,ffn_num_input,ffn_num_hiddens,ffn_num_output,num_layers,dropout=0,use_bias=False):
        super(TransformerEncoderCnn, self).__init__()
        self.num_hidden=num_hiddens
        self.embedding_hiddens=embedding_hiddens
        self.embedding=nn.Embedding(vocab_size,embedding_hiddens)
        self.pos_encoding=PositionalEncoding(embedding_hiddens,dropout)
        self.blks=nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("TransformerEncoderBlock"+str(i),EncoderBlockCnn(key_size,query_size,value_size,num_hiddens,num_heads,
                                                                               norm_shape,ffn_num_input,ffn_num_hiddens,ffn_num_output,dropout,use_bias))
    def forward(self,x,valid_lens):
        x=self.pos_encoding(self.embedding(x)*math.sqrt(self.embedding_hiddens))
        self.attention_weights=[None]*len(self.blks)
        self.attention_weights_cbam_channel = [None] * len(self.blks)
        self.attention_weights_cbam_spatial = [None] * len(self.blks)
        for i,blk in enumerate(self.blks):
            x=blk(x,valid_lens)
            self.attention_weights[i]=blk.attention.attention.attention_weights
            self.attention_weights_cbam_channel[i]=blk.attentionCnn.atttention_channel
            self.attention_weights_cbam_spatial[i]=blk.attentionCnn.atttention_spatial
        return x


class TransformerEncoder(nn.Module):
    def __init__(self,vocab_size,key_size,query_size,value_size,embedding_hiddens,num_hiddens,num_heads,
                 norm_shape,ffn_num_input,ffn_num_hiddens,ffn_num_output,num_layers,dropout=0,use_bias=False):
        super(TransformerEncoder, self).__init__()
        self.num_hidden=num_hiddens
        self.embedding_hiddens=embedding_hiddens
        self.embedding=nn.Embedding(vocab_size,embedding_hiddens)
        self.pos_encoding=PositionalEncoding(embedding_hiddens,dropout)
        self.blks=nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("TransformerEncoderBlock"+str(i),EncoderBlock(key_size,query_size,value_size,num_hiddens,num_heads,
                                                                               norm_shape,ffn_num_input,ffn_num_hiddens,ffn_num_output,dropout,use_bias))
    def forward(self,x,valid_lens):
        x=self.pos_encoding(self.embedding(x)*math.sqrt(self.embedding_hiddens))
        self.attention_weights=[None]*len(self.blks)
        for i,blk in enumerate(self.blks):
            x=blk(x,valid_lens)
            self.attention_weights[i]=blk.attention.attention.attention_weights
        return x

class DecoderBlock(nn.Module):
    def __init__(self,key_size,query_size,value_size,num_hiddens,num_heads,
                 norm_shape,ffn_num_input,ffn_num_hiddens,ffn_num_output,i,dropout=0,use_bias=False):
        super(DecoderBlock, self).__init__()
        self.i=i
        self.attention1=MultiHeadAttention(key_size,value_size,query_size,num_heads,num_hiddens,dropout)
        self.addnorm1=AddNorm(norm_shape,dropout)
        self.attention2=MultiHeadAttention(key_size,value_size,query_size,num_heads,num_hiddens,dropout)
        self.addnorm2=AddNorm(norm_shape,dropout)
        self.ffn=PositionWiseFFN(ffn_num_input,ffn_num_output,ffn_num_hiddens)
        self.addnorm3=AddNorm(norm_shape,dropout)
    def forward(self,x,state):
        enc_outputs,enc_valid_lens=state[0],state[1]
        if state[2][self.i] is None:
            key_values=x
        else:
            key_values=torch.cat((state[2][self.i],x),axis=1)
        state[2][self.i]=key_values
        if self.training:
            batch_size,num_steps,_=x.shape
            dec_valid_lens=torch.arange(1,num_steps+1,device=x.device).repeat(batch_size,1)
        else:
            dec_valid_lens=None
        x2=self.attention1(x,key_values,key_values,dec_valid_lens)
        y=self.addnorm1(x,x2)
        y2=self.attention2(y,enc_outputs,enc_outputs,enc_valid_lens)
        z=self.addnorm2(y,y2)
        return self.addnorm3(z,self.ffn(z)),state

class TransformerDecoder(nn.Module):
    def __init__(self,vocab_size,key_size,query_size,value_size,num_hiddens,embedding_hiddens,num_heads,
                 norm_shape,ffn_num_input,ffn_num_hiddens,ffn_num_output,num_layers,dropout=0,use_bias=False):
        super(TransformerDecoder, self).__init__()
        self.num_hidden = num_hiddens
        self.embedding_hiddens = embedding_hiddens
        self.num_layers=num_layers
        self.embedding=nn.Embedding(vocab_size,embedding_hiddens)
        self.pos_encoding=PositionalEncoding(embedding_hiddens,dropout)
        self.blks=nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("TransformerDecoder"+str(i),DecoderBlock(key_size,query_size,value_size,num_hiddens,num_heads,
                                                                          norm_shape,ffn_num_input,ffn_num_hiddens,ffn_num_output,i,dropout))
        self.dense=nn.Linear(ffn_num_output,vocab_size)

    def init_state(self,enc_outputs,enc_valid_lens,*args):
        return [enc_outputs,enc_valid_lens,[None]*self.num_layers]

    def forward(self,x,state):
        x=self.embedding(x) * math.sqrt(self.embedding_hiddens)
        x=self.pos_encoding(x)
        self._attention_weights=[[None]*len(self.blks) for _ in range(2)]
        if not self.training:
            x=x[:,-1:,:]
        for i ,blk in enumerate(self.blks):
            x,state=blk(x,state)
            self._attention_weights[0][i]=blk.attention1.attention.attention_weights
            self._attention_weights[1][i] = blk.attention2.attention.attention_weights
        return self.dense(x),state
    def attention_weights(self):
        return self._attention_weights
class TextCnn(nn.Module):
    def __init__(self,embed_size,kernel_size,num_channels):
        super(TextCnn, self).__init__()
        self.droput=nn.Dropout(0.3)
        self.relu=nn.ReLU()
        self.pool=nn.AdaptiveAvgPool1d(1)
        self.convs=nn.ModuleList()
        for c,k in zip(num_channels,kernel_size):
            self.convs.append(nn.Conv1d(embed_size,c,k))
        self.relu = nn.ReLU()
        self.output = nn.Linear(sum(num_channels), 62)
    def forward(self,x):
        embeding = x.permute(0,2,1)
        encoding=torch.cat([
            torch.squeeze(self.relu(self.pool(conv(embeding))),dim=-1)
            for conv in self.convs],dim=1
        )
        return self.relu(self.output(self.droput(encoding)))

if __name__ == '__main__':
    # x=torch.ones((5,4), dtype=torch.long)
    # x_lens = torch.tensor([1,2,3,4,4])
    # encoder = TransformerEncoder(10,10,10,10,10,10,2,[10],10,10,10,1,0.1)
    # model=Encoder(encoder,4,10,2)
    # print(model(x,x_lens).shape)
    # a=TextCnn(128,[3,4,5],[128,128,128])
    # x=torch.ones(5,6,128)
    # b=a(x)
    # print(b.shape)
    a=EncoderBlockCnn(128,128,128,128,8,[128],128,128,128,0.1)
    b=torch.ones((5,5,128))
    c=torch.tensor([1,2,3,4,5])
    print(a(b,c).shape)