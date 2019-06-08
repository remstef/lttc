#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''

@author: Steffen Remus (@remstef)
'''

import argparse
import time
import os, sys
import math
from tqdm import tqdm
import torch
from torch.utils.data.sampler import BatchSampler, SequentialSampler, RandomSampler
import torchnet

import data
import utils
import modules
import embedding

class LttcPipe(object):

  def __init__(self):
    super(LttcPipe, self).__init__()
    self.pargs = utils.AttributeHolder()

  def prepareSystemArgs(self):
    parser = argparse.ArgumentParser(description='Text classification', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--module', default='ConvKim', type=str, help='Module which should be used for training or testing [ ConvKim (default), ... ].')
    parser.add_argument('--model', default='./savedmodels/model', type=str, help='path to save the final model')
    parser.add_argument('--lang', type=str, default='en', help="Language. Currently supported: en (default), de, fr.")
    parser.add_argument('--serve', action='store_true', help='use this switch to load and serve a model.')
    parser.add_argument('--server', type=str, default='127.0.0.1:8881', help='Serve model on HOST:PORT (default=127.0.0.1:8881). Only used if --serve is activated.')
    parser.add_argument('--train', type=str, default='./data/SMSSpamCollection/train', help="dataset location which should be used for training (e.g. './data/SMSSpamCollection' for full data or './data/SMSSpamCollection/train' for training data only).")
    parser.add_argument('--test', type=str, default='./data/SMSSpamCollection/test', help="dataset location which should be used for testing (e.g. './data/SMSSpamCollection/test'). Can be omitted for training, testing will then be performed on the training data.")
    parser.add_argument('--bert-model', default='bert-base-uncased', type=str, help='Bert pre-trained model that is also used for wordpiece tokenization.')
    parser.add_argument('--epochs', default=100, type=int, help='upper epoch limit')
    parser.add_argument('--optim', default='SGD', type=str, help='type of optimizer (SGD, Adam, Adagrad, ASGD, SimpleSGD)')
    parser.add_argument('--loss', default='NLLLoss', type=str, help='type of loss function to use (NLLLoss, CrossEntropyLoss, MarginLoss, SpreadLoss)')
    parser.add_argument('--emsize-word', default=300, type=int, help='size of word embeddings')
    parser.add_argument('--emsize-posi', default=5, type=int, help='size of the position embeddings')
    parser.add_argument('--maxlength', default=-1, type=int, help='maximum length of a sequence (use -1 for determining the length from the training data)')
    parser.add_argument('--context-window', default=0, type=int, help='size of the moving window of left and right contexts for concatenatation')
    parser.add_argument('--convfilters', default='1024,1024,1024', type=str, help='number of convolution filters to apply')
    parser.add_argument('--convwindows', default='3,4,5', type=str, help='sizes of the moving convolutional window')
    parser.add_argument('--convstrides', default='1,1,1', type=str, help='strides of convolutions')
    parser.add_argument('--conv-activation', default='ReLU', type=str, help='activation function to use after convolutinal layer (ReLU, Tanh)')
    parser.add_argument('--nhid', default=200, type=int, help='size of hidden layer')
    parser.add_argument('--lr', default=.1, type=float, help='initial learning rate')
    parser.add_argument('--lr-decay', default=0.25, type=float, help='decay amount of learning learning rate if no validation improvement occurs')
    parser.add_argument('--wdecay', default=1.2e-6, type=float, help='weight decay applied to all weights')
    parser.add_argument('--l1reg', default=.0, type=float, help='add l1 regularuzation loss')
    parser.add_argument('--clip', default=-1, type=float, help='gradient clipping (set to -1 to avoid clipping)')
    parser.add_argument('--batch-size', default=16, type=int, metavar='N', help='batch size')
    parser.add_argument('--dropout', default=0.5, type=float, help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--seed', default=1111, type=int, help='random seed')
    parser.add_argument('--nlines', default=-1, type=int, metavar='N', help='number of lines to process, -1 for all')
    parser.add_argument('--status-reports', default=3, type=int, metavar='N', help='generate N reports during one training epoch (N=min(N, nbatches))')
    parser.add_argument('--init-emword', default='', type=str, help='path to initial word embedding; emsize-word must match size of embedding')
    parser.add_argument('--fix-emword', action='store_true', help='Specify if the word embedding should be excluded from further training')
    parser.add_argument('--shuffle-samples', action='store_true', help='shuffle samples')
    parser.add_argument('--shuffle-batches', action='store_true', help='shuffle batches')
    parser.add_argument('--cuda', action='store_true', help='use CUDA')
    #parser.add_argument('--testswitch', action='store_true', help='some test switch for quick code debugging')
    return parser

  def parseSystemArgs(self):
    parser = self.prepareSystemArgs()
    args = utils.AttributeHolder(**parser.parse_args().__dict__)
    args.convfilters = [ int(x) for x in filter(lambda x: x, map(lambda x: x.strip(), args.convfilters.strip(' ,').split(','))) ]
    args.convwindows = [ int(x) for x in filter(lambda x: x, map(lambda x: x.strip(), args.convwindows.strip(' ,').split(','))) ]
    args.convstrides = [ int(x) for x in filter(lambda x: x, map(lambda x: x.strip(), args.convstrides.strip(' ,').split(','))) ]
    return args

  def prepareCuda(self, args):
    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
      if not args.cuda:
        print('WARNING: You have a CUDA device, you would probably want to run with --cuda', file=sys.stderr)
    self.pargs.device = torch.device('cuda' if args.cuda else 'cpu')
    return args

  def loadDatasets(self, args):
    nlines = None if args.nlines <= 0 else args.nlines

    trainset = data.LttcDataset(path = args.train,
                                lang = args.lang,
                                nlines = nlines,
                                maxseqlen = args.maxlength,
                                nbos = args.context_window,
                                neos = args.context_window).load().to(self.pargs.device)
    print('train: ' + str(trainset), file=sys.stderr)
    self.freeze_and_save_indices(args, trainset.index, trainset.posiindex, trainset.classindex)
    args.maxseqlentrain = trainset.maxseqlen
    args.nbostrain = trainset.nbos
    args.neostrain = trainset.neos

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
                                 classindex = trainset.classindex).load().to(self.pargs.device)
    print('test: ' + str(testset), file=sys.stderr)

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
    return args


  def loadaddlData(self, args):
    preemb_weights = None
    # load pre embedding
    if args.init_emword:
      # determine type of embedding by checking it's suffix
      if args.init_emword.endswith('bin'):
        preemb = embedding.FastTextEmbedding(args.init_emword, normalize = True).load()
        if args.emsize_word != preemb.dim():
          raise ValueError(f'emsize-word must match embedding size. Expected {args.emsize_word:d} but got {preemb.dim():d}')
      elif args.init_emword.endswith('txt'):
        preemb = embedding.TextEmbedding(args.init_emword, vectordim = args.emsize_word).load(normalize = True)
      elif args.init_emword.endswith('rand'):
        preemb = embedding.RandomEmbedding(vectordim = args.emsize_word)
      else:
        raise ValueError('Type of embedding cannot be inferred.')
      preemb = embedding.Embedding.filteredEmbedding(self.pargs.trainset.index.vocabulary(), preemb, fillmissing = True)
      preemb_weights = torch.Tensor(preemb.weights)
    self.pargs.emweights_word = preemb_weights
    return args


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
    return args


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

    # processing function
    def process(batch_data, istraining):
      targets = batch_data['label']
      outputs, labelweights = model(**batch_data)

      loss = criterion(outputs, targets)

      predictions = self.getpredictions(outputs.data)
      return loss, (batch_data['id'], outputs, predictions, targets)

    print(model, file=sys.stderr)
    print(criterion, file=sys.stderr)
    print(optimizer, file=sys.stderr)

    self.pargs.model = model
    self.pargs.modelprocessfun = process
    self.pargs.criterion = criterion
    self.pargs.optimizer = optimizer
    return args


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


  def freeze_and_save_indices(self, args, wordindex, positionindex, classindex):
    # make sure the directory exists
    os.makedirs(args.model, exist_ok=True)
    wordindex.freeze(silent = True).tofile(os.path.join(args.model, 'ndx_vocab.txt'))
    positionindex.freeze(silent = True).tofile(os.path.join(args.model, 'ndx_position.txt'))
    classindex.freeze(silent = False).tofile(os.path.join(args.model, 'ndx_classes.txt'))


  def savemodel(self, args, epoch):
    # make sure the directory exists
    os.makedirs(args.model, exist_ok=True)

    # save the NN model
    with open(os.path.join(args.model, 'model.pt'), 'wb') as f:
      torch.save(self.pargs.model, f)

    # save the parameters
    args.modelepoch = epoch
    with open(os.path.join(args.model, 'parameters.yml'), 'wt') as f:
      args.dump(f)

  def savepredictions(self, args, ids, logprobs, predictions, targets, scores):
    outfile = os.path.join(args.model, 'model.predictions.tsv')
    assert len(ids) == len(logprobs) == len(predictions) == len(targets), f'Something is wrong, number of samples and number of predicions are different: {len(ids):s} {len(logprobs):s} {len(predictions):s} {len(targets):s}'
    with open(outfile, 'w') as f:
      print('# ' + scores.string.replace('\n', '\n# '), file=f)
      for i in range(len(ids)):
        pred_classlabel = self.pargs.classindex[predictions[i]]
        true_classlabel = self.pargs.classindex[targets[i]]
        correct = int(predictions[i] == targets[i])
        print(f'{ids[i]:d}\t{pred_classlabel:s}\t{true_classlabel:s}\t{correct:d}\t{predictions[i]:d}\t{targets[i]:d}\t{logprobs[i]:}', file=f)


  ###############################################################################
  # Run Pipeline
  ###############################################################################
  def pipeline(self, args):

    def evaluate(args, dloader):
      model = self.pargs.model
      # Turn on evaluation mode which disables dropout.
      model.eval()
      test_loss_batch = torch.zeros(len(dloader))
      ids = []
      predictions = []
      logprobs = []
      targets = []

      with torch.no_grad():
        for batch_i, batch_data in enumerate(tqdm(dloader, ncols=89, desc = 'Test ')):
          loss, (sampleids, outputs, predictions_, targets_) = process(batch_data, False)
          if args.l1reg > 0:
            reg_loss = l1reg(model)
            loss += args.l1reg * reg_loss
          # keep track of some scores
          test_loss_batch[batch_i] = loss.item()
          self.pargs.confusion_meter.add(outputs.data, targets_)
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
      model = self.pargs.model
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
        loss, (_, outputs, batch_predictions, batch_targets) = process(batch_data, True)
        if args.l1reg > 0:
          reg_loss = l1reg(model)
          loss += args.l1reg * reg_loss
        loss.backward()
        self.pargs.optimizer.step()
        # track some scores
        train_loss_batch[batch_i] = loss.item()
        self.pargs.confusion_meter.add(outputs.data, batch_targets)
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
    best_run = utils.AttributeHolder(test_val=0, epoch=0)
    args.status_reports = min(args.status_reports, self.pargs.ntrainbatches)
    self.pargs.report_after_n_samples = math.ceil(self.pargs.ntrainsamples / (args.status_reports+1))
    self.pargs.confusion_meter = torchnet.meter.ConfusionMeter(self.pargs.nclasses, normalized=True)
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
      tqdm.write(self.message_status_endepoch('', epoch, epoch_start_time, self.pargs.optimizer.getLearningRate(), train_loss, test_loss, train_scores, test_scores, best_run))
      if best_run.test_val < test_scores[self.pargs.best_run_test_valname]:
        tqdm.write(f'''  > Saving model and prediction results to '{args.model:s}'...''')
        self.savemodel(args, epoch)
        self.savepredictions(args, test_sampleids, test_logprobs, test_predictions, test_targets, test_scores)
        best_run.test_valname = self.pargs.best_run_test_valname
        best_run.test_val = test_scores[best_run.test_valname]
        best_run.epoch = epoch
        best_run.train_scores_cum = train_scores_cum
        best_run.train_scores = train_scores
        best_run.test_scores = test_scores
        best_run.train_loss = train_loss
        best_run.test_loss = test_loss
        tqdm.write('  > ... Finished saving\n  |')


  def build_and_run(self, args):
    self.args = self.prepareLoader(args)
    self.args = self.buildModel(self.args)
    self.pipeline(self.args)


  def load(self, dirname):
    # load model args
    with open(os.path.join(dirname, 'parameters.yml'), 'rt') as f:
      self.pargs.modelargs = utils.AttributeHolder().load(f)
    print(self.pargs.modelargs, file=sys.stderr)
    # load model
    with open(os.path.join(dirname, 'model.pt'), 'rb') as f:
      self.pargs.model = torch.load(f, map_location=self.pargs.device)
    print(self.pargs.model, file=sys.stderr)
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
                            classindex = classindex).to(self.pargs.device)


  def serve(self, dirname, server):
    self.load(dirname)
    self.pargs.model.eval()

    def predict_sample(text):
      tensors = self.pargs.dset.process_sample(text)
      if not tensors:
        return f"EMPTY OR INVALID INPUT: {text.encode('utf-8')}"
      # add batch dimension
      for t in tensors.values():
        t.unsqueeze_(0)

      outputs, _ = self.pargs.model(**tensors)
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
        self.serve(self.args.model, self.args.server)
      else:
        self.args = self.loadDatasets(self.args)
        self.args = self.loadaddlData(self.args)
        self.args = self.prepareLoader(self.args)
        self.args = self.buildModel(self.args)
        self.pipeline(self.args)
    except (KeyboardInterrupt, SystemExit):
      print('Process cancelled', file=sys.stderr)


if __name__ == '__main__':
  pipe = LttcPipe()
  pipe.run()
