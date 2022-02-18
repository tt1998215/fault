#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：1.20 
@File ：MAE.py
@Author ：ts
@Date ：2022/2/11 21:17 
'''
import torch
from torch import nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self,input_dim,num_hiddens,num_layers):
        super(Encoder, self).__init__()
        # self.embedding=nn.Linear(input_dim,embed_dim)
        self.rnn=nn.LSTM(input_dim,num_hiddens,num_layers,dropout=0.1,batch_first=True)
        self.activation=nn.ReLU()

    def forward(self,x):
        # x=self.embedding(x)
        x=self.activation(x)
        output,state=self.rnn(x)
        return output,state

class Decoder(nn.Module):
    """用于序列到序列学习的循环神经网络解码器"""
    def __init__(self, input_dim, num_hiddens, num_layers,
                 ):
        super(Decoder, self).__init__()
        # self.embedding = nn.Linear(input_dim, embed_dim)
        self.rnn = nn.LSTM(input_dim + num_hiddens, num_hiddens, num_layers,
                          dropout=0.1,batch_first=True)
        self.dense = nn.Linear(num_hiddens, input_dim)
        self.dense2 = nn.Linear(input_dim, 4)
        self.activation=nn.ReLU()

    def init_state(self, enc_outputs, *args):
        return enc_outputs[1]

    def forward(self, X, state):
        # 输出'X'的形状：(batch_size,num_steps,embed_size)
        # X = self.embedding(X)
        # 广播context，使其具有与X相同的num_steps
        context = state[-1][-1].repeat(X.shape[1], 1, 1).permute(1,0,2)
        X_and_context = torch.cat((X, context), 2)
        X_and_context=self.activation(X_and_context)
        output, state = self.rnn(X_and_context, state)
        output = self.dense(output)
        output=self.activation(output)
        # output的形状:(batch_size,num_steps,vocab_size)
        # state[0]的形状:(num_layers,batch_size,num_hiddens)
        output=self.dense2(torch.mean(output[:,-10:,:],dim=1).unsqueeze(1))
        output=F.softmax(output.squeeze(1))
        return output

class EncoderDecoder(nn.Module):
    """The base class for the encoder-decoder architecture.
    Defined in :numref:`sec_encoder-decoder`"""

    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state)
if __name__ == '__main__':
    encoder=Encoder(15,64,2)
    tensor=torch.ones((64,50,15))

    decoder=Decoder(15,64,2)
    net=EncoderDecoder(encoder,decoder)
    out=net(tensor,tensor[:,-30:,])
    print(out)

