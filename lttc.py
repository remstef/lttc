#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''

@author: Steffen Remus (@remstef)
'''

import argparse
import time
import os, sys
import copy
import math
from tqdm import tqdm
import torch
from torch.utils.data.sampler import BatchSampler, SequentialSampler, RandomSampler

import data
import utils
import modules
import embedding

class LttcPipe(object):

  def __init__(self):
    super(LttcPipe, self).__init__()
    self.pargs = utils.AttributeHolder()

  def prepareSystemArgs(self):
    
    class StoreAction(argparse._StoreAction):
      def __call__(self, parser, namespace, values, option_string=None):
        super(StoreAction, self).__call__(parser, namespace, values, option_string)
        setattr(namespace, '_explicit_', getattr(namespace, '_explicit_', [ '_explicit_' ]) + [ self.dest ])

    class StoreTrueAction(argparse._StoreTrueAction):
      def __call__(self, parser, namespace, values, option_string=None):
        super(StoreTrueAction, self).__call__(parser, namespace, values, option_string)
        setattr(namespace, '_explicit_', getattr(namespace, '_explicit_', [ '_explicit_' ]) + [ self.dest ])
    
    parser = argparse.ArgumentParser(description='Text classification', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--train', type=str, default='./data/SMSSpamCollection/train', help="dataset location which should be used for training (e.g. './data/SMSSpamCollection' for full data or './data/SMSSpamCollection/train' for training data only).", action=StoreAction)
    parser.add_argument('--test', type=str, default='./data/SMSSpamCollection/test', help="dataset location which should be used for testing (e.g. './data/SMSSpamCollection/test'). Can be omitted for training, testing will then be performed on the training data.", action=StoreAction)
    parser.add_argument('--configs', type=str, nargs='*', help='Run multiple configs. One config is the path to a configuration in yaml format to use.', action=StoreAction)
    parser.add_argument('--module', default='ConvKim', type=str, help='Module which should be used for training or testing [ ConvKim (default), ... ].', action=StoreAction)
    parser.add_argument('--model', default='./savedmodels/model', type=str, help='path to save the final model', action=StoreAction)
    parser.add_argument('--lang', type=str, default='en', help="Language. Currently supported: en (default), de, fr.", action=StoreAction)
    parser.add_argument('--serve', help='use this switch to load and serve a model.', action=StoreTrueAction)
    parser.add_argument('--server', type=str, default='127.0.0.1:8881', help='Serve model on HOST:PORT (default=127.0.0.1:8881). Only used if --serve is activated.', action=StoreAction)
    parser.add_argument('--bert-model', default='bert-base-uncased', type=str, help='Bert pre-trained model that is also used for wordpiece tokenization.', action=StoreAction)
    parser.add_argument('--epochs', default=100, type=int, help='upper epoch limit', action=StoreAction)
    parser.add_argument('--optim', default='SGD', type=str, help='type of optimizer (SGD, Adam, Adagrad, ASGD, SimpleSGD)', action=StoreAction)
    parser.add_argument('--optimsched', default=None, type=str, help='type of scheduler for adjustment of the the learning rate during training [ None (default), CosineAnnealingLR, ...)', action=StoreAction)
    parser.add_argument('--loss', default='NLLLoss', type=str, help='type of loss function to use [ NLLLoss (default), CrossEntropyLoss, MarginLoss, SpreadLoss, ... ]', action=StoreAction)
    parser.add_argument('--emsize-word', default=300, type=int, help='size of word embeddings', action=StoreAction)
    parser.add_argument('--emsize-posi', default=5, type=int, help='size of the position embeddings', action=StoreAction)
    parser.add_argument('--maxlength', default=-1, type=int, help='maximum length of a sequence (use -1 for determining the length from the training data)', action=StoreAction)
    parser.add_argument('--context-window', default=0, type=int, help='size of the moving window of left and right contexts for concatenatation', action=StoreAction)
    parser.add_argument('--convfilters', default=[1024,1024,1024], type=int, nargs='*', help='number of convolution filters to apply', action=StoreAction)
    parser.add_argument('--convwindows', default=[3,4,5], type=int, nargs='*', help='sizes of the moving convolutional window', action=StoreAction)
    parser.add_argument('--convstrides', default=[1,1,1], type=int, nargs='*', help='strides of convolutions', action=StoreAction)
    parser.add_argument('--conv-activation', default='ReLU', type=str, help='activation function to use after convolutinal layer (ReLU, Tanh)', action=StoreAction)
    parser.add_argument('--nhid', default=200, type=int, help='size of hidden layer', action=StoreAction)
    parser.add_argument('--lr', default=.1, type=float, help='initial learning rate', action=StoreAction)
    parser.add_argument('--lr-decay', default=0.25, type=float, help='decay amount of learning learning rate if no validation improvement occurs', action=StoreAction)
    parser.add_argument('--wdecay', default=1.2e-6, type=float, help='weight decay applied to all weights', action=StoreAction)
    parser.add_argument('--l1reg', default=.0, type=float, help='add l1 regularuzation loss', action=StoreAction)
    parser.add_argument('--clip', default=-1, type=float, help='gradient clipping (set to -1 to avoid clipping)', action=StoreAction)
    parser.add_argument('--batch-size', default=16, type=int, metavar='N', help='batch size', action=StoreAction)
    parser.add_argument('--dropout', default=0.5, type=float, help='dropout applied to layers (0 = no dropout)', action=StoreAction)
    parser.add_argument('--seed', default=1111, type=int, help='random seed', action=StoreAction)
    parser.add_argument('--nlines', default=-1, type=int, metavar='N', help='number of lines to process, -1 for all', action=StoreAction)
    parser.add_argument('--status-reports', default=3, type=int, metavar='N', help='generate N reports during one training epoch (N=min(N, nbatches))', action=StoreAction)
    parser.add_argument('--init-emword', default='', type=str, help='path to initial word embedding; emsize-word will be overwritten with the size of the embedding.', action=StoreAction)
    parser.add_argument('--fix-emword', help='Specify if the word embedding should be excluded from further training', action=StoreTrueAction)
    parser.add_argument('--shuffle-samples', help='shuffle samples', action=StoreTrueAction)
    parser.add_argument('--shuffle-batches', help='shuffle batches', action=StoreTrueAction)
    parser.add_argument('--cuda', help='use CUDA', action=StoreTrueAction)
    # parser.add_argument('--testswitch', help='some test switch for quick code debugging', action=StoreTrueAction)
    return parser

  def parseSystemArgs(self):
    parser = self.prepareSystemArgs()
    args = utils.AttributeHolder(**parser.parse_args().__dict__)
    return args
  
  def parseArgsFromConfigfile(self, currentargs, fname, keep):
    try:
      currentargs.load(fname, keep=currentargs._explicit_) # read and overwrite args with args from config
    except FileNotFoundError:
      print(f'File {fname} does not exist!', file=sys.stderr)
      return None
    currentargs.configval=fname
    # make args with path information relative to the path in that the configuration was found
    if not 'model' in currentargs._explicit_:
      currentargs.modelval=currentargs.model
      currentargs.model=os.path.join(os.path.dirname(fname), currentargs.model)
    if not 'init_emword' in currentargs._explicit_ and currentargs.init_emword and currentargs.init_emword.startswith('.'):
      currentargs.init_emwordval=currentargs.init_emword
      currentargs.init_emword=os.path.join(os.path.dirname(fname), currentargs.init_emword)
    return currentargs
  
  def prepareCuda(self, args):
    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
      if not args.cuda:
        print('WARNING: You have a CUDA device, you would probably want to run with --cuda', file=sys.stderr)
    self.pargs.device = torch.device('cuda' if args.cuda else 'cpu')
    return args

  def loadDatasets(self, args):
    
    def updateArgsFromTrainingData(trainset):
      args.maxseqlentrain = trainset.maxseqlen
      args.nbostrain = trainset.nbos
      args.neostrain = trainset.neos
      args.maxseqlen_berttrain = trainset.maxseqlen_bert

    # if already set, don't load it again, but only if the pretrained embedding still matches
    if self.pargs.has('trainset') and self.pargs.trainset.path == args.train and self.pargs.has('emweights_word_name') and self.pargs.emweights_word_name == args.init_emword:
      updateArgsFromTrainingData(self.pargs.trainset)
      return
    
    nlines = None if args.nlines <= 0 else args.nlines
    trainset = data.LttcDataset(path = args.train,
                                lang = args.lang,
                                nlines = nlines,
                                maxseqlen = args.maxlength,
                                nbos = args.context_window,
                                neos = args.context_window,
                                bert_model = args.bert_model).load().to(self.pargs.device)
    print('train: ' + str(trainset), file=sys.stderr)

    self.freeze_and_save_indices(args, trainset.index, trainset.posiindex, trainset.classindex)
    if args.init_emword: # if a pretrained embedding is used, e.g. Bert or fasttext, then add also test set words to the index (and thus embedding)
      trainset.index.unfreeze()

    updateArgsFromTrainingData(trainset)

    testset = None
    if args.test:
      testset = data.LttcDataset(path = args.test,
                                 lang = args.lang,
                                 nlines = nlines,
                                 maxseqlen = trainset.maxseqlen,
                                 index = trainset.index,
                                 nbos = trainset.nbos,
                                 neos = trainset.neos,
                                 posiindex = trainset.posiindex,
                                 classindex = trainset.classindex,
                                 bert_model = trainset.bert_tokenizer).load().to(self.pargs.device)
    print('test: ' + str(testset), file=sys.stderr)
    
    if not trainset.index.frozen: # if a pretrained embedding is used
      self.freeze_and_save_indices(args, trainset.index)

    self.pargs.trainset = trainset
    self.pargs.testset = testset
    self.pargs.classindex = trainset.classindex
    self.pargs.emword_pad_idx=trainset.padidx
    self.pargs.best_run_test_valname = 'F'

    self.pargs.maxseqlen = trainset.maxseqlen
    self.pargs.ntrainsamples = len(trainset)
    self.pargs.ntoken = len(trainset.index)
    self.pargs.npositions = len(trainset.posiindex)
    self.pargs.nclasses = len(self.pargs.classindex)



  def loadaddlData(self, args):
    preemb_weights = None
    # load pre embedding
    if args.init_emword:
      if self.pargs.has('emweights_word') and self.pargs.has('emweights_word_name') and self.pargs.emweights_word_name == args.init_emword:
        args.emsize_word = self.pargs.emweights_word.size(1)
        return
      # determine type of embedding by checking it's suffix
      if args.init_emword.endswith('bin'):
        preemb = embedding.FastTextEmbedding(args.init_emword, normalize = True).load()
      elif args.init_emword.startswith('bert-'):
        preemb = embedding.BertEmbedding(args.init_emword, normalize = True).load()
      elif args.init_emword.endswith('txt'):
        preemb = embedding.TextEmbedding(args.init_emword, vectordim = 300).load(normalize = True, skipheader=True) # TODO: this works for google word2vec embeddings in text format but not for arbitrary text embeddings (skipheader & dimension)
      elif args.init_emword.endswith('rand'):
        preemb = embedding.RandomEmbedding(vectordim = args.emsize_word)
      else:
        raise ValueError('Type of embedding cannot be inferred.')
      preemb = embedding.Embedding.filteredEmbedding(self.pargs.trainset.index.vocabulary(), preemb, fillmissing = True)
      print(f'Resetting emsize-word to {preemb.dim()}.', file=sys.stderr)
      args.emsize_word = preemb.dim()
      preemb_weights = torch.Tensor(preemb.weights)
    self.pargs.emweights_word_name = args.init_emword
    self.pargs.emweights_word = preemb_weights


  def prepareLoader(self, args):
    __ItemSampler = RandomSampler if args.shuffle_samples else SequentialSampler
    train_loader = torch.utils.data.DataLoader(self.pargs.trainset, batch_sampler = utils.ShufflingBatchSampler(BatchSampler(__ItemSampler(self.pargs.trainset), batch_size=args.batch_size, drop_last = False), shuffle = args.shuffle_batches, seed = args.seed), num_workers = 0)

    test_loader = None
    if self.pargs.testset:
      test_loader = torch.utils.data.DataLoader(self.pargs.testset, batch_sampler = BatchSampler(__ItemSampler(self.pargs.testset), batch_size=args.batch_size, drop_last = False), num_workers = 0)

    print(__ItemSampler.__name__, file=sys.stderr)
    print('Shuffle training batches: ', args.shuffle_batches, file=sys.stderr)

    self.pargs.trainloader = train_loader
    self.pargs.testloader = test_loader
    self.pargs.ntrainbatches = len(train_loader)


  def buildModel(self, args):
    '''
      Build the model, loss, optimizer and processing function
    '''
    # model
    if self.pargs.has('modelclass'):
      modelclass__ = self.pargs.modelclass
    else:
      if not hasattr(modules, args.module):
        raise ValueError( f"Unknown module '{args.module}'.")
      modelclass__ = getattr(modules, args.module)
      self.pargs.modelclass = modelclass__

    model = modelclass__(**args.__dict__, **self.pargs.__dict__)
    model = model.to(self.pargs.device)

    # loss
    if not hasattr(torch.nn, args.loss):
      raise ValueError( '''Unknown loss criterion `%s`.''' % args.loss)
    criterion = getattr(torch.nn, args.loss)()

    # optimizer
    if args.optim == 'SimpleSGD':
      Optimizer__ = utils.SimpleSGD
    else:
      if not hasattr(torch.optim, args.optim):
        raise ValueError( '''Invalid option `%s` for 'optimizer' was supplied.''' % args.optim)
      Optimizer__ = getattr(torch.optim, args.optim)
    optimizer = utils.createWrappedOptimizerClass(Optimizer__)(model.parameters(), lr =args.lr, clip=args.clip, weight_decay=args.wdecay)
    
    # scheduler, e.g. in order to control the learning rate of the optimizer
    scheduler = optimizer
    if args.optimsched:
      if not hasattr(torch.optim.lr_scheduler, args.optimsched):
        raise ValueError( f"Unknown scheduler '{args.optimsched}'.")
      scheduler = getattr(torch.optim.lr_scheduler, args.optimsched)(optimizer, args.epochs)

    # processing function
    def process(batch_data, istraining):
      targets = batch_data['label']
      outputs, labelweights, predictions = self.apply(model, batch_data)
      loss = criterion(outputs, targets)
      return loss, (batch_data['id'], outputs, predictions, targets)

    print(model, file=sys.stderr)
    num_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total number of model parameters: {num_total_params:d}', file=sys.stderr)
    print(criterion, file=sys.stderr)
    print(optimizer, file=sys.stderr)
    print(args.optimsched, file=sys.stderr)

    self.pargs.modelinstance = model
    self.pargs.modelprocessfun = process
    self.pargs.modelcriterion = criterion
    self.pargs.modeloptimizer = optimizer
    self.pargs.modeloptimizingscheduler = scheduler

    
  def apply(self, model, dict_batch_data):
    outputs, labelweights = model(**dict_batch_data)
    predictions = self.getpredictions(outputs.data)
    return outputs, labelweights, predictions


  def getpredictions(self, batch_logprobs):
    return batch_logprobs.max(dim=1)[1]


  def getscores(self, targets, predictions, extended=False):
    return self.pargs.trainset.compute_scores(targets, predictions, extended=extended)


  def message_status_interval(self, message, report_i, nreports, epoch, max_epoch, report_batch_start, report_batch_end, nbatches, batch_size, report_sample_start, report_sample_end, nsamples, report_batch_start_time, train_loss_interval, train_loss_cumulated, scores_int, scores_cum):
    return '''\
  |   {:s}
  |   Report {:d} / {:d}
  |   +-- Epoch {:d}/{:d} | Batch {:d}-{:d}/{:d} ({:d}) | Sample {:d}-{:d}/{:d} | ms/batch {:5.2f}
  |   +-- Training loss {:.10f}   ({:.10f})
  |   +--   {:s}
  |   +-- ( {:s} )
  |\
  '''.format(
        message,
        report_i,
        nreports,
        epoch,
        max_epoch,
        report_batch_start,
        report_batch_end,
        nbatches,
        batch_size,
        report_sample_start,
        report_sample_end,
        nsamples,
        ((time.time() - report_batch_start_time) * 1000) / (report_batch_end - report_batch_start),
        train_loss_interval,
        train_loss_cumulated,
        scores_int.string.replace('\n', '\n' + ' '*10),
        scores_cum.string.replace('\n', '\n' + ' '*10))


  def message_status_endepoch(self, message, epoch, epoch_start_time, learning_rate, train_loss, test_loss, train_scores, test_scores, best_before):
    trainscoreline = train_scores.string
    testscoreline  = test_scores.string
    bestbeforeline = '---' if not best_before.epoch else f'''E {best_before.epoch:d} | L (train) {best_before.train_loss:.4f} | L (test) {best_before.test_loss:.4f} | {best_before.test_valname} (test) {best_before.test_val:6.3f}'''
    return '''\
  |
  |{:s}
  | Epoch {:03d} took {:06.2f}s
  |   +-- Learing rate   {:s}
  |   +-- Loss (train)   {:.10f}
  |   +-- Loss (test)    {:.10f}
  |   +-- Scores (train) {:s}
  |   +-- Scores (test)  {:s}
  |   +-- Best (before)  {:s}
  |{:s}
  |
  |\
  '''.format(
        '=' * 88,
        epoch,
        (time.time() - epoch_start_time),
        '{:10.6f}'.format(learning_rate).strip(),
        train_loss,
        test_loss,
        trainscoreline.replace('\n', '\n' + ' '*25),
        testscoreline.replace('\n', '\n' + ' '*25),
        bestbeforeline,
        '=' * 88)
  
  
  def message_status_endeval(self, test_scores):
    testscoreline  = test_scores.string
    return '''\
  |
  |{:s}
  |   +-- Scores (eval) {:s}
  |{:s}\
  '''.format(
        '=' * 88,
        testscoreline.replace('\n',  '\n  |' + ' '*21),
        '=' * 88)


  def freeze_and_save_indices(self, args, wordindex, positionindex, classindex):
    # make sure the directory exists
    os.makedirs(args.model, exist_ok=True)
    wordindex.freeze(silent = True).tofile(os.path.join(args.model, 'ndx_vocab.txt'))
    positionindex.freeze(silent = True).tofile(os.path.join(args.model, 'ndx_position.txt'))
    classindex.freeze(silent = False).tofile(os.path.join(args.model, 'ndx_classes.txt'))


  def savemodel(self, args, epoch, message='', suffix='-final'):
    # make sure the directory exists
    os.makedirs(args.model, exist_ok=True)

    # save the NN model
    with open(os.path.join(args.model, f'model{suffix}.pt'), 'wb') as f:
      torch.save(self.pargs.modelinstance, f)

    # save the parameters and the status message
    args.modelepoch = epoch
    args.dump(dest=os.path.join(args.model, f'parameters{suffix}.yml'))
    with open(os.path.join(args.model, f'status{suffix}.txt'), 'wt') as f:
      m = '\n'.join(filter(lambda l: len(l) > 1, map(lambda l: l.strip(), message.split('\n'))))
      print(m, file=f)


  def savepredictions(self, args, ids, logprobs, predictions, targets, scores, suffix='-final'):
    outfile = os.path.join(args.model, f'testpredictions{suffix}.tsv')
    assert len(ids) == len(logprobs) == len(predictions) == len(targets), f'Something is wrong, number of samples and number of predicions are different: {len(ids):s} {len(logprobs):s} {len(predictions):s} {len(targets):s}'
    with open(outfile, 'wt') as f:
      print('# ' + scores.string.replace('\n', '\n# '), file=f)
      for i in range(len(ids)):
        pred_classlabel = self.pargs.classindex[predictions[i]]
        true_classlabel = self.pargs.classindex[targets[i]]
        correct = int(predictions[i] == targets[i])
        print(f'{ids[i]:d}\t{pred_classlabel:s}\t{true_classlabel:s}\t{correct:d}\t{predictions[i]:d}\t{targets[i]:d}\t{logprobs[i]:}', file=f)
        
        
  def load(self, dirname, suffix=''):
    # load model args
    self.pargs.modelargs = utils.AttributeHolder().load(f'parameters{suffix}.yml')
    print(self.pargs.modelargs, file=sys.stderr)
    # load model
    with open(os.path.join(dirname, f'model{suffix}.pt'), 'rb') as f:
      self.pargs.modelinstance = torch.load(f, map_location=self.pargs.device)
    print(self.pargs.modelinstance, file=sys.stderr)
    # load indices
    wordindex = utils.Index.fromfile(os.path.join(dirname, 'ndx_vocab.txt')).freeze(silent=True)
    positionindex = utils.Index.fromfile(os.path.join(dirname, 'ndx_position.txt')).freeze(silent=True)
    classindex = utils.Index.fromfile(os.path.join(dirname, 'ndx_classes.txt')).freeze(silent=True)
    self.pargs.indices = (wordindex, positionindex, classindex)
    [ print(index, file=sys.stderr) for index in self.pargs.indices ]
    # prepare dataset for pre-processing
    self.pargs.dset = data.LttcDataset(path = None,
                            lang = self.pargs.modelargs.lang,
                            maxseqlen = self.pargs.modelargs.maxseqlentrain,
                            index = wordindex,
                            nbos = self.pargs.modelargs.nbostrain,
                            neos = self.pargs.modelargs.neostrain,
                            posiindex = positionindex,
                            classindex = classindex,
                            bert_model = self.pargs.modelargs.bert_model,
                            maxseqlen_bert = self.pargs.modelargs.maxseqlen_berttrain).to(self.pargs.device)


  ###############################################################################
  # Run Pipeline
  ###############################################################################
  def pipeline(self, args):

    def evaluate(args, dloader):
      model = self.pargs.modelinstance
      # Turn on evaluation mode which disables dropout.
      model.eval()
      test_loss_batch = torch.zeros(len(dloader))
      ids = []
      predictions = []
      logprobs = []
      targets = []

      with torch.no_grad():
        for batch_i, batch_data in enumerate(tqdm(dloader, ncols=89, desc = 'Test ')):
          loss, (sampleids, outputs, predictions_, targets_) = process(batch_data, istraining=False)
          if args.l1reg > 0:
            reg_loss = l1reg(model)
            loss += args.l1reg * reg_loss
          # keep track of some scores
          test_loss_batch[batch_i] = loss.item()
          ids.extend(sampleids.tolist())
          logprobs.extend(outputs.data.tolist())
          predictions.extend(predictions_.tolist())
          targets.extend(targets_.tolist())
      test_loss = test_loss_batch.mean()
      return test_loss, ids, logprobs, predictions, targets, test_loss_batch

    def l1reg(model):
      # add l1 regularization
      reg_loss = 0
      for param_i, param in enumerate(model.parameters()):
        if param is None:
          continue
        reg_loss += torch.functional.F.l1_loss(param, target=torch.zeros_like(param), size_average=False)
      reg_loss /= (param_i+1)
      return reg_loss

    def train(args):
      model = self.pargs.modelinstance
      # Turn on training mode which enables dropout.
      model.train()

      train_loss_batch = torch.zeros(len(self.pargs.trainloader))
      sample_i = 0
      report_i = 0
      report_interval_begin_sample = 0
      report_interval_begin_batch = 0
      predictions = []
      targets = []

      for batch_i, batch_data in enumerate(tqdm(self.pargs.trainloader, ncols=89, desc='Train')):
        batch_start_time = time.time()
        model.zero_grad()
        loss, (_, outputs, batch_predictions, batch_targets) = process(batch_data, istraining=True)
        if args.l1reg > 0:
          reg_loss = l1reg(model)
          loss += args.l1reg * reg_loss
        loss.backward()
        self.pargs.modeloptimizingscheduler.step()
        # track some scores
        train_loss_batch[batch_i] = loss.item()
        predictions.extend(batch_predictions.tolist())
        targets.extend(batch_targets.tolist())
        sample_i += batch_targets.size(0)

        if ((sample_i - report_interval_begin_sample) // self.pargs.report_after_n_samples) > 0:
          cur_loss = train_loss_batch[report_interval_begin_batch:(batch_i+1)].mean()
          cur_scores = self.getscores(targets[report_interval_begin_sample:], predictions[report_interval_begin_sample:])
          cum_scores = self.getscores(targets, predictions)
          tqdm.write(self.message_status_interval('*** training status ***', report_i+1, args.status_reports, epoch, args.epochs, report_interval_begin_batch, batch_i+1, self.pargs.ntrainbatches, args.batch_size, report_interval_begin_sample, len(targets), self.pargs.ntrainsamples, batch_start_time, cur_loss, train_loss_batch.mean(), cur_scores, cum_scores))
          report_interval_begin_sample = len(targets)
          report_interval_begin_batch = batch_i+1
          report_i += 1

      train_loss = train_loss_batch.mean()
      return train_loss, predictions, targets, train_loss_batch

    ###
    # Run pipeline
    ###
    best_run = utils.AttributeHolder(test_val=float('-inf'), epoch=0)
    args.status_reports = min(args.status_reports, self.pargs.ntrainbatches)
    self.pargs.report_after_n_samples = math.ceil(self.pargs.ntrainsamples / (args.status_reports+1))
    process = self.pargs.modelprocessfun
    for epoch in tqdm(range(1,args.epochs+1), ncols=89, desc = 'Epochs'):
      epoch_start_time = time.time()
      train_loss_cum, train_predictions_cum, train_targets_cum, _ = train(args)
      train_scores_cum = self.getscores(train_targets_cum, train_predictions_cum)

      # test training set
      train_loss, train_sampleids, train_logprobs, train_predictions, train_targets, _ = evaluate(args, self.pargs.trainloader)
      train_scores = self.getscores(train_targets, train_predictions)
      # test test set
      if self.pargs.testset:
        test_loss, test_sampleids, test_logprobs, test_predictions, test_targets, _ = evaluate(args, self.pargs.testloader)
        test_scores = self.getscores(test_targets, test_predictions, extended=True)
      else:
        test_loss, test_sampleids, test_logprobs, test_predictions, test_targets = train_loss, train_sampleids, train_logprobs, train_predictions, train_targets
        test_scores = train_scores

      # print scores
      status_message = self.message_status_endepoch('', epoch, epoch_start_time, self.pargs.modeloptimizer.getLearningRate(), train_loss, test_loss, train_scores, test_scores, best_run)
      tqdm.write(status_message)
      if best_run.test_val < test_scores[self.pargs.best_run_test_valname]:
        tqdm.write(f'''  > Saving model and prediction results to '{args.model:s}'...''')
        self.savemodel(args, epoch, status_message, suffix='')
        self.savepredictions(args, test_sampleids, test_logprobs, test_predictions, test_targets, test_scores, suffix=f'')
        best_run.test_valname = self.pargs.best_run_test_valname
        best_run.test_val = test_scores[best_run.test_valname]
        best_run.epoch = epoch
        best_run.train_scores_cum = train_scores_cum
        best_run.train_scores = train_scores
        best_run.test_scores = test_scores
        best_run.train_loss = train_loss
        best_run.test_loss = test_loss
        tqdm.write('  > ... Finished saving\n  |')
    # save final model and scores
    tqdm.write(f'''  > Saving final model and prediction results to '{args.model:s}'...''')
    self.savemodel(args, epoch, status_message, suffix='-final')
    self.savepredictions(args, test_sampleids, test_logprobs, test_predictions, test_targets, test_scores, suffix='-final')
    tqdm.write('  > ... Finished saving\n  |')


  def build_and_run(self, args):
    self.prepareLoader(args)
    self.buildModel(args)
    self.pipeline(args)
    

  def serve(self, dirname, server):
    self.load(dirname)
    self.pargs.modelinstance.eval()
    self.pargs.dset.cache_device_tensors = False

    def predict_sample(text):
      tensors = self.pargs.dset.process_sample(text)
      if not tensors:
        return f"EMPTY OR INVALID INPUT: {text.encode('utf-8')}"
      # add batch dimension
      for t in tensors.values():
        t.unsqueeze_(0)

      outputs, _ = self.pargs.modelinstance(**tensors)
      predictions = self.getpredictions(outputs.data)

      # remove batch dimension by using only first entry
      prediction = predictions[0].data
      label = self.pargs.dset.classindex.getWord(prediction)
      outputs_sample_numpy = outputs.data[0].cpu().numpy()
      explicit = ' '.join(map(lambda t: f'{self.pargs.dset.classindex.getWord(t[0])}:{t[1]:.4f}', enumerate(outputs_sample_numpy)))
      print(explicit, file=sys.stderr)

      return f'{label}\t{outputs[0][prediction]}\t{explicit}'

    # test prediction
    predict_sample('hello brave new world')

    # start the server
    h, p = server.split(':')
    utils.create_tcp_server(HOST=h, PORT=int(p), linehandler=predict_sample)


  def run(self):
    try:
      self.args = self.parseSystemArgs()
      self.args = self.prepareCuda(self.args)
      if self.args.serve:
        torch.manual_seed(self.args.seed)
        self.serve(self.args.model, self.args.server)
      else:
        if self.args.configs:
          config_args = list(map(lambda cfg: (self.parseArgsFromConfigfile(copy.copy(self.args), cfg, keep=self.args._explicit_), cfg), self.args.configs))
        else:
          config_args = [ (self.args, None) ]
        for args, config_name in config_args:
          if not args:
            print(f'Config {config_name} does not exist. Skipping!', file=sys.stderr)
            continue
          if os.path.isfile(os.path.join(args.model, 'model-final.pt')):
            print(f"Model file already exists, skipping '{args.model}' (config_name).", file=sys.stderr)
            continue
          try:
            torch.manual_seed(args.seed)
            self.loadDatasets(args)
            self.loadaddlData(args)
            self.prepareLoader(args)
            self.buildModel(args)
            self.pipeline(args)
          except (KeyboardInterrupt, SystemExit):
            print('Process cancelled -- skipping config', file=sys.stderr)
    except (KeyboardInterrupt, SystemExit):
      print('Process cancelled', file=sys.stderr)


if __name__ == '__main__':
  pipe = LttcPipe()
  pipe.run()
