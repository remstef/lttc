# -*- coding: utf-8 -*-
'''

@author: Steffen Remus (@remstef)
'''

import sys, os
import random
import pickle
import torch.utils.data


class RequiredParam(object):
  @staticmethod
  def check(argdict, caller=None):
    for k, v in argdict.items():
      if v is RequiredParam:
        raise Exception(f''''{k:s}' is a required parameter but not provided for '{caller:s}'.''')


class AttributeHolder(object):
  def __init__(self, **kwargs):
    [ self.__setitem__(k,v) for k,v in kwargs.items() ]
  def __repr__(self):
    return f'{self.__class__.__name__:s}({self.__dict__.__repr__():s})'
  def __setitem__(self, key, value):
    return setattr(self, key, value)
  def __getitem__(self, key):
    if not hasattr(self, key):
      return None
    return getattr(self, key)
  def has(self, key):
    if hasattr(self, key):
      return self.__getitem__(key) is not None
    return False


class Index(object):

  def __init__(self, initwords = [], unkindex = None):
    self.id2w = []
    self.w2id = {}
    self.frozen = False
    self.unkindex = unkindex
    if initwords is not None:
      for w in initwords:
        self.add(w)

  def add(self, word):
    word = str(word)
    if word in self.w2id:
      return self.w2id[word]
    if self.frozen:
      if not self.silentlyfrozen:
        msg = f'Failed adding `{word}` to index, using `unk`. Cause: Index can not be altered, it is already frozen.'
        print(msg, file=sys.stderr)
        #raise ValueError(msg)
      else:
        return self.unkindex
    idx = len(self.id2w)
    self.w2id[word] = idx
    self.id2w.append(word)
    return idx

  def size(self):
    return len(self.w2id)

  def hasId(self, idx):
    return idx >= 0 and idx < len(self.id2w)

  def hasWord(self, word):
    return str(word) in self.w2id

  def getWord(self, index):
    return self.id2w[index]

  def getId(self, word):
    try:
      return self.w2id[str(word)]
    except KeyError:
      return self.unkindex

  def tofile(self, fname):
    with open(fname, 'w', encoding='utf8') as f:
      lines = map(lambda w: f'{w[0]:d}:{str(w[1]):s}\n', enumerate(self.id2w))
      f.writelines(lines)

  def freeze(self, silent = False):
    self.frozen = True
    self.silentlyfrozen = silent
    return self

  def vocabulary(self):
    return self.id2w

  def __contains__(self, key):
    if isinstance(key, str):
      return self.hasWord(key)
    return self.hasId(key)

  def __getitem__(self, key):
    # return the id if we get a word
    if isinstance(key, str):
      return self.getId(key)
    # return the word if we get an id, lets assume that 'key' is some kind of number, i.e. int, long, ...
    if not hasattr(key, '__iter__'):
      return self.getWord(key)
    # otherwise recursively apply this method for every key in an iterable
    return map(lambda k: self[k], key)

  def __iter__(self):
    return self.id2w.__iter__()

  def __len__(self):
    return self.size()

  def __repr__(self):
    subseq = self.id2w[:min(10,len(self.id2w))]
    return 'Index ([\n  {}\n{}]:{:d})'.format(
        '\n  '.join(map(lambda tup: f'{tup[0]:4d}: {tup[1]}', enumerate(subseq))),
        '     ...\n' if len(self.id2w) > len(subseq) else '',
        len(self.id2w))

  @staticmethod
  def fromfile(fname):
    index = Index()
    with open(fname, 'r', encoding='utf8') as f:
      for i, line in enumerate(f):
        w = line.rstrip().split(':', 1)
        assert int(w[0]) == i
        index.id2w.append(w[1])
        index.w2id[w[1]] = i
    return index


class FunctionModule(torch.nn.Module):

  def __init__(self, fun):
    super(FunctionModule, self).__init__()
    self.fun = fun

  def forward(self, x):
    y = self.fun(x)
    return y


class Attention(torch.nn.Module):
  '''
  attn = Attention(100)
  x = Variable(torch.randn(16,30,100))
  attn(x).size() == (16,100)
  '''
  def __init__(self, attention_size):
    super(Attention, self).__init__()
    self.attention = self.new_parameter(attention_size, 1)

  def forward(self, x_in):
    # after this, we have (batch, dim1) with a diff weight per each cell
    attention_score = torch.matmul(x_in, self.attention).squeeze()
    attention_score = torch.functional.softmax(attention_score).view(x_in.size(0), x_in.size(1), 1)
    scored_x = x_in * attention_score

    # now, sum across dim 1 to get the expected feature vector
    condensed_x = torch.sum(scored_x, dim=1)

    return condensed_x

  @staticmethod
  def new_parameter(*size):
      out = torch.nn.Parameter(torch.FloatTensor(*size))
      torch.nn.init.xavier_normal(out)
      return out

class RandomBatchSampler(torch.utils.data.sampler.BatchSampler):

  def __init__(self, *args, **kwargs):
    super(RandomBatchSampler, self).__init__(*args, **kwargs)

  def __iter__(self):
    batches = list(super().__iter__())
    random.shuffle(batches)
    for batch in batches:
      yield batch

class ShufflingBatchSampler(torch.utils.data.sampler.BatchSampler):

  def __init__(self, batchsampler, shuffle = True, seed = 10101):
    self.batchsampler = batchsampler
    self.shuffle = True
    self.seed = seed
    self.numitercalls = -1

  def __iter__(self):
    self.numitercalls += 1
    batches = self.batchsampler.__iter__()
    if self.shuffle:
      batches = list(batches)
      random.seed(self.seed+self.numitercalls)
      random.shuffle(batches)
    for batch in batches:
      yield batch

  def __len__(self):
    return len(self.batchsampler)


class EvenlyDistributingSampler(torch.utils.data.sampler.BatchSampler):
  '''
  Test:

    [[chr(i+ord('a')) for i in batch] for batch in EvenlyDistributingSampler(SequentialSampler(list(range(25))), batch_size=4, drop_last=True)]

  '''
  def __init__(self, sampler, batch_size, drop_last, *args, **kwargs):
    super(EvenlyDistributingSampler, self).__init__(sampler, batch_size, drop_last, *args, **kwargs)
    if not drop_last:
      raise NotImplementedError('Drop last is not yet implemented for `EvenlyDistributingSampler`.')
    self.sampler = sampler
    self.batch_size = batch_size

  def __iter__(self):
    # Starting from sequential data, batchify arranges the dataset into columns.
    # For instance, with the alphabet as the sequence and batch size 4, we'd get
    # ┌ a g m s ┐
    # │ b h n t │
    # │ c i o u │
    # │ d j p v │
    # │ e k q w │
    # └ f l r x ┘.
    # These columns are treated as independent by the model, which means that the
    # dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
    # batch processing.
    #  def batchify(data, bsz):
    #    # Work out how cleanly we can divide the dataset into bsz parts.
    #    nbatch = data.size(0) // bsz
    #    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    #    data = data.narrow(0, 0, nbatch * bsz)
    #    # Evenly divide the data across the bsz batches.
    #    data = data.view(bsz, -1).t().contiguous()
    #    return data.to(device)

    # get_batch subdivides the source data into chunks of length args.bptt.
    # If source is equal to the example output of the batchify function, with
    # a bptt-limit of 2, we'd get the following two Variables for i = 0:
    # ┌ a g m s ┐ ┌ b h n t ┐
    # └ b h n t ┘ └ c i o u ┘
    # Note that despite the name of the function, the subdivison of data is not
    # done along the batch dimension (i.e. dimension 1), since that was handled
    # by the batchify function. The chunks are along dimension 0, corresponding
    # to the seq_len dimension in the LSTM.
    #  def get_batch(source, i):
    #    seq_len = min(args.bptt, len(source) - 1 - i)
    #    data = source[i:i+seq_len]
    #    target = source[i+1:i+1+seq_len].view(-1)
    #    return data, target

    # tests:
    # data = torch.Tensor([i for i in range(ord('a'),ord('z')+1)]).long()
    # [xyz = chr(i) for i in [for r in data]]
    #

    # each sampler returns indices, use those indices
    data = torch.LongTensor(list(self.sampler))
    nbatch = data.size(0) // self.batch_size
    data = data.narrow(0, 0, nbatch * self.batch_size)
    data = data.view(self.batch_size, -1).t() # this is important!

    for row_as_batch in data:
      yield row_as_batch.tolist()


class SimpleSGD(torch.optim.Optimizer):

  def __init__(self, params, *args, lr=RequiredParam, **kwargs):
    if lr is not RequiredParam and lr < 0.0:
      raise ValueError('Invalid learning rate: {}'.format(lr))
    defaults = dict(lr=lr)
    super(SimpleSGD, self).__init__(params, defaults)

  def step(self, closure=None):
    loss = None
    if closure is not None:
      loss = closure()

    for group in self.param_groups:
      for p in group['params']:
        if p.grad is None:
          continue
        d_p = p.grad.data
        p.data.add_(-group['lr'], d_p)

    return loss


def createWrappedOptimizerClass(optimizer_clazz):
  '''
  Provide methods in order to
    - get the current learning rate of an optimizer
    - adjust the learning rate by a factor
    - perform clipping of gradients before a step
  '''
  class Wrapped(optimizer_clazz):
    def __init__(self,  *args, clip = 0.2, **kwargs):
      super(Wrapped, self).__init__(*args, **kwargs)
      self.clip = clip
    def getLearningRate(self):
      lr = [group['lr'] for group in self.param_groups]
      return lr[0] if len(lr) == 1 else lr
    def adjustLearningRate(self, factor=None):
      for group in self.param_groups:
        newlr = group['lr'] * factor
        group['lr'] = newlr
    def step(self, closure=None):
      loss = None
      if closure is not None:
        loss = closure()
      groups = self.param_groups
      if self.clip is not None and self.clip > 0:
        for group in groups:
          torch.nn.utils.clip_grad_norm_(group['params'], self.clip)
      super(Wrapped, self).step(closure=None)
      return loss
    def __repr__(self):
      return f'{optimizer_clazz.__name__}:{super(Wrapped, self).__repr__().replace(os.linesep,os.linesep+"  ")}(clip: {self.clip:.3f})'

  return Wrapped

def pad(x, length, padval):
  y = torch.ones((length,)).long() * padval
  y[:min(len(x), length)] = x[:min(len(x), length)]
  return y

def makeOneHot(X, maxval):
  X_one_hot = X.new_zeros(*X.size(), maxval).float()
  X_one_hot = X_one_hot.scatter(-1, X.unsqueeze(-1), 1.)
  return X_one_hot

def makeBow(X_one_hot):
   X_bow = X_one_hot.sum(dim=1)
   return X_bow

def norm_safe(x, dim=-1, eps=1e-15, keepdim=False):
  squared_norm = (x ** 2).sum(dim=dim, keepdim=keepdim)
  safe_norm = squared_norm + eps
  return torch.sqrt(safe_norm)

def squash_safe(x, dim=-1, eps=1e-15):
  squared_norm = (x ** 2).sum(dim=dim, keepdim=True)
  safe_norm = torch.sqrt(squared_norm + eps)
  scale = squared_norm / (1. + squared_norm)
  unit_vector = x / safe_norm
  return scale * unit_vector

def argmax_noisy(x, dim=None, keepdim=False, eps=1e-15):
  noise = x.clone().uniform_(-eps, eps)
  noisy_x = noise + x
  return torch.argmax(noisy_x, dim=dim, keepdim=keepdim)

def pickle_dump_large(df, fout, tqdm):
  # write in chunks of 128 MBytes
  max_bytes = 2**27 - 1
  print('Packing data...', file=sys.stderr)
  bytes_out = pickle.dumps(df)
  with open(fout, 'wb') as f:
    for idx in tqdm(range(0, len(bytes_out), max_bytes), ncols=89):
      f.write(bytes_out[idx:idx+max_bytes])

def pickle_read_large(fin, tqdm):
  # read in chunks of 128 MBytes
  max_bytes = 2**27 - 1
  bytes_in = bytearray(0)
  input_size = os.path.getsize(fin)
  with open(fin, 'rb') as f:
    for _ in tqdm(range(0, input_size, max_bytes), ncols=89):
      bytes_in += f.read(max_bytes)
  print('Unpacking data...', file=sys.stderr)
  df = pickle.loads(bytes_in)
  return df


def merge_context(seq, s):
  '''
  in:  seq = batch_size x seqence x features
  in:    s = left and right context size
  out:       batch_size x seqence-2*s x (features x (sx2+1))

  concatenate sliding windows (proper padding of sequences beforehand is expected)

  x = torch.Tensor(list(range(3*4*5))).view(3,4,5)
  tensor([[[ 0.,  1.,  2.,  3.,  4.],
           [ 5.,  6.,  7.,  8.,  9.],
           [10., 11., 12., 13., 14.],
           [15., 16., 17., 18., 19.]],

          [[20., 21., 22., 23., 24.],
           [25., 26., 27., 28., 29.],
           [30., 31., 32., 33., 34.],
           [35., 36., 37., 38., 39.]],

          [[40., 41., 42., 43., 44.],
           [45., 46., 47., 48., 49.],
           [50., 51., 52., 53., 54.],
           [55., 56., 57., 58., 59.]]])

  y = merge_context(x, 1)
  tensor([[[ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12., 13., 14.],
           [ 5.,  6.,  7.,  8.,  9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19.]],

          [[20., 21., 22., 23., 24., 25., 26., 27., 28., 29., 30., 31., 32., 33., 34.],
           [25., 26., 27., 28., 29., 30., 31., 32., 33., 34., 35., 36., 37., 38., 39.]],

          [[40., 41., 42., 43., 44., 45., 46., 47., 48., 49., 50., 51., 52., 53., 54.],
           [45., 46., 47., 48., 49., 50., 51., 52., 53., 54., 55., 56., 57., 58., 59.]]])
  '''
  n = s*2+1
  return torch.cat([seq[:,i:seq.size(1)-(n-i-1),:] for i in range(n)],dim=2)

def create_tcp_server(HOST='127.0.0.1', PORT=8881, linehandler=lambda l: l):

    import socketserver
    class TCPHandler(socketserver.StreamRequestHandler):
      rbufsize = -1 #1024 # default -1
      wbufsize = 0 # default 0
      timeout  = None # 5 default None
      disable_nagle_algorithm = False # default False

      def handle(self):
        for i, bytesin in enumerate(self.rfile):
          print(f'{self.client_address[0]}: {bytesin}', file=sys.stderr)
          textin = bytesin.decode('utf-8')
          textout = linehandler(textin)
          if not textout.endswith('\n'):
            textout += '\n'
          bytesout = textout.encode('utf-8')
          self.wfile.write(bytesout)

    class Server(socketserver.ThreadingTCPServer):
      # block_on_close = True
      # timeout = None
      # address_family = socket.AF_INET
      # socket_type = socket.SOCK_STREAM
      allow_reuse_address = True  # much faster rebinding
      daemon_threads = True # Ctrl-C will cleanly kill all spawned thread

      def __init__(self, *args, **kwargs):
        super(Server, self).__init__(*args, **kwargs)

      def handle_timeout(self):
        print('Timeout', file = sys.stderr)

    #socketserver.ThreadingTCPServer((HOST, PORT), TCPHandler).serve_forever()
    srvr = Server((HOST, PORT), TCPHandler)
    # start
    print(f'Staring TCP Server, listening on {HOST} {PORT}.', file=sys.stderr)
    srvr.serve_forever()


class SimpleRepl(object):
  '''
  Provide some simple REPL functionality
  '''
  def __init__(self, evaluator=lambda cmd: print("You entered '%s'." % cmd), PS1 = '>> '):
    self.ps1 = PS1
    self.evaluator = evaluator

  def read(self):
    return input(self.ps1)

  def evaluate(self):
    command = self.read()
    return self.evaluator(command)

  def run(self):
    while True:
      try:
        self.evaluate()
      except KeyboardInterrupt:
        print('\nBye Bye\n')
        break
