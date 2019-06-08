# -*- coding: utf-8 -*-
'''

@author: Steffen Remus (@remstef)
'''

import sys
import math
import torch
from pytorch_pretrained_bert import BertModel
from utils import RequiredParam, merge_context

##
#
##
class ConvKim(torch.nn.Module):
  '''

  '''

  def __init__(self,
             *args,
             ntoken = RequiredParam,
             nclasses = RequiredParam,
             maxseqlen = RequiredParam,
             npositions = RequiredParam,
             emsize_word = RequiredParam,
             emsize_posi = RequiredParam,
             context_window = 0,
             convfilters = [1024,1024,1024],
             convwindows = [3,4,5],
             convstrides = [1,1,1],
             dropout = 0.5,
             conv_activation = 'ReLU',
             emweights_word=None,
             fix_emword=False,
             emword_pad_idx=None,
             routingiter=3,
             testswitch=False,
             **kwargs):

    super(ConvKim, self).__init__()
    RequiredParam.check(locals(), self.__class__.__name__)

    # parameters
    self.cs = context_window
    self.li = maxseqlen-(2*context_window)
    self.fi = (2*context_window+1) * (emsize_word + 2 * emsize_posi) # size of the feature vector for words
    self.fo = list(map(lambda i: math.ceil((self.li - (convwindows[i]-1)) / convstrides[i]), range(len(convfilters))))

    # activation functions and dropout
    self.drop = torch.nn.Dropout(dropout)
    self.softmax = torch.nn.LogSoftmax(dim=1)
    # activation function for convolutions
    if not conv_activation in ['ReLU', 'Tanh']:
      raise ValueError( '''Invalid option `%s` for 'conv-activation'.''' % conv_activation)
    convact_class = getattr(torch.nn, conv_activation)

    # layers
    self.word_embeddings = torch.nn.Embedding(ntoken, emsize_word, padding_idx=emword_pad_idx)
    self.posi_embeddings = torch.nn.Embedding(npositions, emsize_posi)
    self.convl = torch.nn.ModuleList([
        torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels = 1,
                out_channels = convfilters[i],
                kernel_size = (convwindows[i], self.fi),
                stride = (convstrides[i], 1),
                bias=True),
            convact_class(),
            self.drop,
            torch.nn.MaxPool2d((self.fo[i], 1))
        ) for i in range(len(convfilters)) ])
    self.lin = torch.nn.Linear(sum(convfilters), nclasses)

    # initialization actions
    self.init_weights(emweights_word, fix_emword)

  def init_weights(self, emweights_word, fix_emword):
    initrange = 0.1
    if emweights_word is None:
      self.word_embeddings.weight.data.uniform_(-initrange, initrange)
    else:
      assert emweights_word.size() == self.word_embeddings.weight.size(), f'Size clash emwords supplied weights: {emweights_word.size()}, expected {self.word_embeddings.weight.size()}'
      self.word_embeddings.load_state_dict({'weight': emweights_word})
    if fix_emword:
      self.word_embeddings.weight.requires_grad = False
    self.posi_embeddings.weight.data.uniform_(-initrange, initrange)

  def forward(self, *args, seq=RequiredParam, seqlen=RequiredParam, seqposi=RequiredParam, seqposi_rev=RequiredParam, **kwargs):

    # seq = batch_size x max_seq_length (padded) : sentence
    # seqlen = batch_size x seq_length
    (bsize, seqlen) = seq.size()

    ## BEGIN:
    s = self.word_embeddings(seq)
    p1 = self.posi_embeddings(seqposi)
    p2 = self.posi_embeddings(seqposi_rev)

    # concatenate word embedding with positional embedding, w = batch_size x seq_length x (wemsize+2xpemsize)
    w = torch.cat((s, p1, p2), dim=2)
    w = self.drop(w)

    # concatenate embeddings with their context embeddings in a sliding window fashion, w = batch_size x  seq_length-windowsize//2-1 x (windowsize x (wemsize+2xpemsize))
    if self.cs > 0:
      w = merge_context(w, self.cs)

    # convolution
    w.unsqueeze_(1) # add `channel` dimension; needed for conv: w = batch_size x 1 x seq_length x nfeatures

    wz = [ m(w) for m in self.convl ]
    wz = [ wzi.view(*wzi.size()[:2]) for wzi in wz ] # remove trailing singular dimensions (f: batch_size x numfilters x 1 x 1 => batch_size x numfilters)

    # x = torch.cat((x1,x2, x3), 1)
    z = torch.cat(wz, dim=1)
    z = self.drop(z)

    y = self.lin(z)
    y = self.drop(y)

    logprobs = self.softmax(y)

    return logprobs, 0



class BertSeqFT(torch.nn.Module):

  def __init__(self,
               *args,
               bert_model = RequiredParam,
               nclasses = RequiredParam,
               testswitch=False,
               **kwargs):

    super(BertSeqFT, self).__init__()

    RequiredParam.check(locals(), self.__class__.__name__)
    
    print(f"Loading bert model '{bert_model}'.", file=sys.stderr)
    self.bertmodel = BertModel.from_pretrained(bert_model)
    self.bertmodel_size = 768
    self.linear = torch.nn.Linear(self.bertmodel_size, nclasses)
    self.softmax = torch.nn.LogSoftmax(dim=1)

  def forward(self, *args, seq_bert=RequiredParam, **kwargs):
    # seq_bert = batch_size x max_seq_length (padded) : sentence
    (batch_size, seqlen) = seq_bert.size()
    
    segments = torch.zeros((batch_size, seqlen)).long().to(seq_bert.device)    
    s, _ = self.bertmodel(seq_bert, segments, output_all_encoded_layers=False)
    s = s[:,0,:].view(batch_size, -1)
    o = self.softmax(self.linear(s))
    
    return o, 0
  
  

class BertSeqNoFT(torch.nn.Module):

  def __init__(self,
               *args,
               bert_model = RequiredParam,
               nclasses = RequiredParam,
               testswitch=False,
               **kwargs):

    super(BertSeqNoFT, self).__init__()

    RequiredParam.check(locals(), self.__class__.__name__)
    
    print(f"Loading bert model '{bert_model}'.", file=sys.stderr)
    self.bertmodel = BertModel.from_pretrained(bert_model)
    self.bertmodel.eval()
    self.bertmodel_size = 768
    self.linear = torch.nn.Linear(self.bertmodel_size, nclasses)
    self.softmax = torch.nn.LogSoftmax(dim=1)

  def forward(self, *args, seq_bert=RequiredParam, **kwargs):
    # seq_bert = batch_size x max_seq_length (padded) : sentence
    (batch_size, seqlen) = seq_bert.size()
    
    segments = torch.zeros((batch_size, seqlen)).long().to(seq_bert.device)
    with torch.no_grad():
      s, _ = self.bertmodel(seq_bert, segments, output_all_encoded_layers=False)
    s = s[:,0,:].view(batch_size, -1)
    o = self.softmax(self.linear(s))
    
    return o, 0