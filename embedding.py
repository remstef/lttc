# -*- coding: utf-8 -*-
'''

@author: Steffen Remus (@remstef)
'''

import numpy as np
import sys
from pyfasttext import FastText
from utils import Index

class Embedding(object):

  def __init__(self, weights, index, normalize = False):
    assert weights.shape[0] == len(index), f'expected {weights.shape[0]:d} but got {len(index):d}. Weights: {str(weights.shape):s}'
    self.normalize = normalize
    self.vdim = weights.shape[1]
    self.index = index
    self.weights = weights
    self.invindex = None

  def getVector(self, word):
    if not self.containsWord(word):
      print("'%s' is unknown." % word, file = sys.stderr)
      return np.zeros(self.vdim)
    idx = self.index.getId(word)
    return self.weights[idx]

  def containsId(self, idx):
    return self.index.hasId(idx)

  def containsWord(self, word):
    return self.index.hasWord(word)

  def vocabulary(self):
    return self.id2w

  def dim(self):
    return self.vdim

  @staticmethod
  def filteredEmbedding(vocabulary, embedding, fillmissing = True):
    index = Index()
    weights = []
    if fillmissing:
      rv = RandomEmbedding(embedding.dim())
    for w in vocabulary:
      if index.hasWord(w):
        continue
      if embedding.containsWord(w):
        index.add(w)
        weights.append(embedding.getVector(w))
      elif fillmissing:
        index.add(w)
        weights.append(rv.getVector(w))
    weights = np.array(weights, dtype = np.float32)
    return Embedding(weights, index)


class RandomEmbedding(Embedding):

  def __init__(self, vectordim = 300):
    self.index = Index()
    self.vdim = vectordim
    self.data = np.zeros((0, self.vdim), dtype = np.float32)
    self.invindex = None

  def getVector(self, word):
    if not self.index.hasWord(word):
      # create random vector
      v = np.random.rand(self.vdim).astype(np.float32)
      # normalize
      length = np.linalg.norm(v)
      if length == 0:
        length += 1e-6
      v = v / length
      # add
      idx = self.index.add(self.id2w)
      self.data = np.vstack((self.data, v))
      assert idx == len(self.data)
      if self.invindex is not None:
        del self.invindex
        self.invindex = None
      return v
    idx = self.index.getId(word)
    return self.data[idx]

  def containsWord(self, word):
    return True

  def vocabulary(self):
    return self.index.vocbulary()

  def dim(self):
    return self.vdim


class FastTextEmbedding(Embedding):

  def __init__(self, binfile, normalize = False):
    self.file = binfile
    self.vdim = -1
    self.normalize = normalize

  def load(self):
    print('Loading fasttext model.')
    self.ftmodel = FastText()
    self.ftmodel.load_model(self.file)
    self.vdim = len(self.ftmodel['is'])
    print('Finished loading fasttext model.')
    return self

  def getVector(self, word):
    return self.ftmodel.get_numpy_vector(word, normalized = self.normalize)

  def search(self, q, topk = 4):
    raise NotImplementedError()

  def wordForVec(self, v):
    word, sim = self.ftmodel.words_for_vector(v)[0]
    return word, sim

  def containsWord(self, word):
    return True

  def vocabulary(self):
    return self.ftmodel.words

  def nearest_neighbors(self, term, n=1000):
    return self.ftmodel.nearest_neighbors(term, n)

  def all_nearest_neighbors(self, term):
    return self.nearest_neighbors(term, len(self.vocabulary()))

  def dim(self):
    return self.vdim


class TextEmbedding(Embedding):

  def __init__(self, txtfile, sep = ' ', vectordim = 300):
    self.file = txtfile
    self.vdim = vectordim
    self.separator = sep

  def load(self, skipheader = True, nlines = sys.maxsize, normalize = False):
    self.index = Index()
    print('Loading embedding from %s' % self.file)
    data_ = []
    with open(self.file, 'r', encoding='utf-8', errors='ignore') as f:
      if skipheader:
        f.readline()
      for i, line in enumerate(f):
        if i >= nlines:
          break
        try:
          line = line.strip()
          splits = line.split(self.separator)
          word = splits[0]
          if self.index.hasWord(word):
            continue
          coefs = np.array(splits[1:self.vdim+1], dtype=np.float32)
          if normalize:
            length = np.linalg.norm(coefs)
            if length == 0:
              length += 1e-6
            coefs = coefs / length
          if coefs.shape != (self.vdim,):
            continue
          idx = self.index.add(word)
          data_.append(coefs)
          assert idx == len(data_)
        except Exception as err:
          print('Error in line %d' % i, sys.exc_info()[0], file = sys.stderr)
          print(' ', err, file = sys.stderr)
          continue
    self.data = np.array(data_, dtype = np.float32)
    del data_
    return self

  def getVector(self, word):
    if not self.containsWord(word):
      print("'%s' is unknown." % word, file = sys.stderr)
      v = np.zeros(self.vdim)
      v[0] = 1
      return v
    idx = self.index.getId(word)
    return self.data[idx]

  def search(self, q, topk = 4):
    if len(q.shape) == 1:
      q = np.matrix(q)
    if q.shape[1] != self.vdim:
      print('Wrong shape, expected %d dimensions but got %d.' % (self.vdim, q.shape[1]), file = sys.stderr )
      return
    D, I = self.invindex.search(q, topk) # D = distances, I = indices
    return ( I, D )

  def wordForVec(self, v):
    idx, dist = self.search(v, topk=1)
    idx = idx[0,0]
    dist = dist[0,0]
    sim = 1. - dist
    word = self.index.getWord(idx)
    return word, sim

  def containsWord(self, word):
    return self.index.hasWord(word)

  def vocabulary(self):
    return self.index.vocabulary()

  def dim(self):
    return self.vdim
