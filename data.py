# -*- coding: utf-8 -*-
'''

@author: Steffen Remus (@remstef)
'''
import sys
import os
import numpy as np
from tqdm import tqdm
import pandas
import torch
import torch.utils
import torch.utils.data
from utils import Index, AttributeHolder, pad, pickle_read_large, pickle_dump_large
import sklearn.datasets
import sklearn.metrics

tqdm.pandas(ncols=89)
spacynlps = {}

def importSpacy(lang='en'):
  if lang == 'de':
    import de_core_news_sm as spacymodel
  elif lang == 'fr':
    import fr_core_news_sm as spacymodel
  else: ## use 'en'
    import en_core_web_sm as spacymodel
  return spacymodel

def getSpacyNLP(spacymodel):
  global spacynlps
  if spacymodel in spacynlps:
    return spacynlps[spacymodel]
  spacynlps[spacymodel] = spacymodel.load()
  return spacynlps[spacymodel]

'''

'''
class LttcDataset(torch.utils.data.Dataset):

  def __init__(self, path=None, lang='en', nlines=None, maxseqlen=None, index = None, nbos = 0, neos = 1, posiindex = None, classindex = None):
    super(LttcDataset, self).__init__()
    self.path = path
    self.maxseqlen = maxseqlen
    self.nbos = max(0, nbos)
    self.neos = max(1, neos)
    self.index = index if index is not None else Index()
    self.padidx = self.index.add('<pad>')
    self.bosidx = self.index.add('<s>')
    self.eosidx = self.index.add('</s>')
    self.index.unkindex = self.index.add('<unk>')
    self.classindex = classindex if classindex is not None else Index()
    self.classindex.unkindex = 0
    self.posiindex = posiindex if posiindex is not None else Index()
    self.nlines = nlines
    self.device = torch.device('cpu')
    self.deviceTensor = torch.LongTensor().to(self.device) # create tensor on device, which can be used for copying
    self.lang = lang
    self.spacy_model = importSpacy(self.lang)
    self.samples = pandas.DataFrame(columns = [ 'id', 'filename', 'rawdata', 'spacydata', 'seqbow', 'seq', 'seqlen', 'seq_recon', 'pseq', 'pseq_rev', 'label', 'labelid' ])

  def process_sample(self, text):
    rawdata = text.strip()
    if len(rawdata) == 0:
      return None
    df = pandas.DataFrame({'rawdata': [ rawdata ], 'label': 'UNK'})
    spacynlp = getSpacyNLP(self.spacy_model)
    df['spacydata'] = df.rawdata.apply(spacynlp)
    # process
    #df = df.progress_apply(self.transform_data_row, axis=1)
    df = df.apply(self.transform_data_row, axis=1)
    df['seqbow'] = df.seq.apply(lambda s: s.new_zeros(len(self.index)).scatter(dim=0, index=s[self.nbos:s.size(0)-self.neos], value=1)) # create bag of word representation w/o bos and eos
    # pad
    df['seq'] = df.seq.apply(lambda s: pad(s, self.maxseqlen, self.padidx))
    df['seqlen'] = df.seqlen.apply(lambda l: min(l, self.maxseqlen))
    # prepare positional sequences
    df = df.apply(self.prepare_positional, axis=1)
    # reconstructed sequence for debugging purposes
    df['seq_recon'] = df.seq.apply(lambda t: np.array(list(self.index[t.tolist()])))
    df['id'] = self.samples.shape[0]
    self.samples = pandas.concat([self.samples, df], axis=0, sort=False, copy=False)
    return self.__getitem__(df.iloc[0].id)

  def preload_file(self, filename):
    # prepare processed filename
    processed_file = f'{filename}__{self.spacy_model.__name__}__{self.spacy_model.__version__}.pkl'
    if self.nlines:
      processed_file = processed_file + f'_{self.nlines:d}'

    # if file exists load samples from there
    if os.path.isfile(processed_file):
      # load preprocessed file if it exists
      tqdm.write(f"Loading preprocessed data from '{processed_file}' ...", file=sys.stderr)
      samples = self.load_processed_samples(processed_file, tqdm)
    else:
      samples = pandas.DataFrame(columns = ['filename', 'label', 'rawdata'])
      # do some preprocessing if preprocessed file does not exist
      tqdm.write(f'Loading data from {filename}...', file=sys.stderr)

      label = os.path.basename(os.path.dirname(filename))
      tqdm.write(f"Reading '{filename}'", file=sys.stderr)
      with open(filename, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f.readlines()):
          if self.nlines and self.nlines <= i:
            break
          samples.loc[len(samples)] = {'filename':f'{filename}:{i}', 'label':label, 'rawdata': line}
      # filter lines that have length zero
      samples.rawdata = samples.rawdata.apply(str.strip)
      samples = samples[samples.rawdata.apply(len) > 0]
      # apply spacy
      tqdm.write('Applying spacy...', file=sys.stderr)
      nlp = getSpacyNLP(self.spacy_model)
      samples['spacydata'] = samples.progress_apply(lambda r: nlp(r.rawdata), axis=1)
      tqdm.write(f"Saving preprocessed data to '{processed_file}' ...", file=sys.stderr)
      self.save_processed_samples(samples, processed_file, tqdm)
    return samples


  def preload(self):
    samples_sets = []
    # path/{train,test}/classlabel/*
    # path/{train,test}/samples__{spacy_modelname}__{spacy_version}.pkl
    for r, d, f in os.walk(self.path, followlinks=True):
      # only if the current directory has no subdirectories, i.e. leaf directories
      if len(d) == 0:
        # for each txt file
        for file in f:
          if file.endswith('.txt'):
            fname = os.path.join(r, file)
            samples_i = self.preload_file(fname)
            samples_sets.append(samples_i)
    samples = pandas.concat(samples_sets, axis=0, sort=False, copy=False)
    return samples

  def load(self):
    samples = self.preload()

    tqdm.write('Preparing data...', file=sys.stderr)
    samples = samples.progress_apply(self.transform_data_row, axis=1)
    samples['seqbow'] = samples.seq.apply(lambda s: s.new_zeros(len(self.index)).scatter(dim=0, index=s[self.nbos:s.size(0)-self.neos], value=1)) # create bag of word representation w/o bos and eos
    # pad
    if not self.maxseqlen or self.maxseqlen < 0:
      self.maxseqlen = samples.seqlen.max()
    samples['seq'] = samples.seq.progress_apply(lambda s: pad(s, self.maxseqlen, self.padidx))
    samples['seqlen'] = samples.seqlen.progress_apply(lambda l: min(l, self.maxseqlen))
    samples = samples[samples.seqlen > (self.nbos + self.neos)] # filter empty samples
    # prepare positional sequences
    samples = samples.progress_apply(self.prepare_positional, axis=1)
    # reconstructed sequence for debugging purposes
    samples['seq_recon'] = samples.seq.progress_apply(lambda t: np.array(list(self.index[t.tolist()])))
    self.samples = pandas.concat([self.samples, samples], axis=0, sort=False, copy=False)
    return self

  def preprocess_text(self, spacydoc):
    d = spacydoc
    d = filter(lambda t : len(t.text) > 0, d)
    #d = filter(lambda t : t.is_alpha and not t.is_stop, d)
    d = map(lambda t : t.text, d)
    d = list(d)
    return d

  def transform_data_row(self, row):
    d = row.spacydata
    d = self.preprocess_text(d)
    seq = list(map(lambda t: self.index.add(t), d))
    # add sentence begin and sentence end markers
    for i in range(self.nbos):
      seq.insert(0, self.bosidx)
    for i in range(max(self.neos, 1)):
      seq.append(self.eosidx)
    seq = torch.LongTensor(seq)
    row['seq'] = seq
    row['seqlen'] = seq.size(0)
    row['labelid'] = self.classindex.add(row['label'])
    row['id'] = hash(seq)
    return row

  def fixindex(self, i, wid, n):
    if wid == self.bosidx:
      return 1
    if wid == self.eosidx:
      return n
    return i+1

  def prepare_positional(self, row):
    s = row.seq
    n = row.seqlen
    ps = torch.arange(n)
    ps.apply_(lambda i: self.fixindex(i, s[i], n))
    ps_rev = -(n+1-ps)
    ps.apply_(lambda i: self.posiindex.add(i))
    ps_rev.apply_(lambda i: self.posiindex.add(i))

    # pad
    padix = self.posiindex.add(self.maxseqlen+1)
    padix_rev = self.posiindex.add(-(self.maxseqlen+1))
    ps = pad(ps, self.maxseqlen, padix)
    ps_rev = pad(ps_rev, self.maxseqlen, padix_rev)
    row['pseq'] = ps
    row['pseq_rev'] = ps_rev
    return row

  def save_processed_samples(self, samples, filename, tqdm):
    #pickle_dump_large(samples, processed_file, tqdm)
    samples.to_pickle(filename)

  def load_processed_samples(self, filename, tqdm):
    #pickle_read_large(processed_file, tqdm)
    return pandas.read_pickle(filename)

  def __len__(self):
    return self.samples.shape[0]

  def __getitem__(self, index):
    r = self.samples.iloc[index]
    s  = r.seq
    sl = r.seqlen
    sb = r.seqbow
    labelid = r.labelid
    sp = r.pseq
    sp_rev = r.pseq_rev
    d = self.deviceTensor
    return \
      d.new_tensor(index), d.new_tensor(r.id), \
      d.new_tensor(s), d.new_tensor(sl), d.new_tensor(sb), \
      d.new_tensor(sp), d.new_tensor(sp_rev), \
      d.new_tensor(labelid)

  def compute_scores(self, true_label_ids, predicted_label_ids, *args, **kwargs):
    # standard scores
    scores = AttributeHolder(
        A = sklearn.metrics.accuracy_score(true_label_ids, predicted_label_ids),
        P = sklearn.metrics.precision_score(true_label_ids, predicted_label_ids, average='macro'),
        R = sklearn.metrics.recall_score(true_label_ids, predicted_label_ids, average='macro'),
        F = sklearn.metrics.f1_score(true_label_ids, predicted_label_ids, average='macro')
        )
    scoreitems = list(enumerate(vars(scores).items()))
    scores.string = ' | '.join(['{:s}{:s} {:6.4f}'.format('' if (i+1) % 5 != 0 else '\n', k, v) for i, (k, v) in scoreitems]).replace(' | \n', '\n')
    return scores

  def cpu(self):
    return self.to(torch.device('cpu'))

  def cuda(self):
    return self.to(torch.device('cuda'))

  def to(self, device):
    print(f"{self.__class__.__name__:s}: Sending new tensors to `{device}`.", file=sys.stderr)
    self.device = device
    self.deviceTensor = self.deviceTensor.to(device)
    return self

  def __repr__(self):
    return f'''\
{self.__class__.__name__:s} (
  path: {self.path}
  maxseqlen: {self.maxseqlen:d}
  wordindex: {self.index}
  posiindex: {self.posiindex}
  classindex: {self.classindex}
  device: {self.device}
  nsamples: {self.samples.shape[0]:d}
  sample[0]: {self.samples.iloc[0].seq_recon if self.samples.shape[0] > 0 else '--'}
)\
'''
