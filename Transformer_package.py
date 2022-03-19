import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=200, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        # create constant 'pe' matrix with values dependant on 
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = \
                math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
 
    
    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        #add constant to embedding
        seq_len = x.size(1)
        pe = Variable(self.pe[:,:seq_len], requires_grad=False)
        if x.is_cuda:
            pe.cuda()
        x = x + pe
        return self.dropout(x)

class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
    def forward(self, x):
        return self.embed(x)


class Transformer(nn.Module):
    def __init__(self, d_model=512, enc_layer=6, dec_layer=6):
        super(Transformer, self).__init__()
        self.enc_layer = enc_layer
        self.dec_layer = dec_layer
        self.d_model = d_model
        self.pe = PositionalEncoder(self.d_model)
        self.embed = Embedder(75, 512)
        encoded_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=8)
        self.encoder = nn.TransformerEncoder(encoded_layer, num_layers=self.enc_layer)
        decoded_layer = nn.TransformerDecoderLayer(d_model=self.d_model, nhead=8)
        self.decoder = nn.TransformerDecoder(decoded_layer, num_layers=self.dec_layer)

    def forward(self, x, txt):
        print('txt: ',txt.shape)
        x = self.pe(x)
        x = self.encoder(x)
        txt = self.embed(txt)
        txt = self.pe(txt)
        x = self.decoder(txt, x)

        return x