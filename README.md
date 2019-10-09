# lttc - eL double Tee See

Supervised textclassification Demo using BERT and CNN with PyTorch.
The implementation is suitable for different classification document scenarios. Sequences per line of text are classified.

The Convolutional Neural Networks based upon the ideas of (Kim, 2014). Further reading: [Understanding Convolutional Neural Networks for NLP](http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/).

The models have by default a softmax layer as ouput layer. So multi class is working out-of-the-box.

## Installing

For convenience there is a makefile. However, especially during build of cython there could be problems depending on your environment. You might walk through the steps manually if this happens.

Recommendations:

using Conda: as reference I added environment.yaml

Run `make install` and check `install.log` - installs all requirments

For training you might need an embedding for initialization of the embedding layer. 

Run `make embedding` to download the pre-trained embeddings 

The application is written in python3 and tested under Mac and Linux. For training the classifier it's recommended to use a GPU with CUDA >9. For reference this package was tested with `Python 3.7.4 [Clang 4.0.1 (tags/RELEASE_401/final)] :: Anaconda, Inc..`.

The make file downloads the following embeddings:

* simple wikipedia fasttext embedding
* German CC fasttext embedding

On Mac you might need to install libomp via brew.

## Data Structure

Once you unpacked the archiv and ran the installation scripts, you should see some data directories.

* ./embedding: The 'embedding' directory contains the embeddings, this is not a requirement, you can put them anywhere in the filesystem
* ./data: The data directory contains an  example dataset (SMSSpamCollection [2]) for reference, any new dataset should roughly follow the same hierarchical structure.
* ./data/SMSSpamCollection: The structure of the dataset is split into two subdirectories 'train' and 'test'. Each subdirectory contains one directory for each class. The name of the directory is also the classlabel. In the current example dataset this is  'ham' and 'spam', for our use case this could be 'training_description' and 'Other' or any name the comes to your mind. Within each of these directories, should be one or more text files (only '.txt' files will be used) which contain only textsamples for the particular class. The textfiles must be UTF-8 encoded and the content must be one document per line! Each line will be interpreted as a sample.
* ./data/TwentyNewsGroups: Same as above only for multiple classes.

## Commands

The main entry point of the program is 'lttc.py', which can be used for both:

* implements a training pipeline in order to train the model
* can be used to simply classify thext data also for multi label classification.

Run

`python lttc.py --help`

to see the list of parameters. 

### Parameters

Classify:

* `--model`(default='./savedmodels/model', type=str): path to save the final model.
* `--lang` (type=str, default='en): Language. Currently supported: en (default), de, fr.

* `--train` (type=str, default='./data/SMSSpamCollection/train): dataset location which should be used for training (e.g. './data/SMSSpamCollection' for full data or './data/SMSSpamCollection/train' for training data only)
* `--test` (type=str, default='./data/SMSSpamCollection/test): dataset location whichshould be used for testing (e.g. './data/SMSSpamCollection/test'). Can be omitted for training, testing will then be performed on the training data.
* `--epochs` (default=100, type=int): upper epoch limit. This should be interrupted if loss and accuracy do not improve, best modell will be saved.
*`--optim` (default='SGD', type=str): type of optimizer (SGD, Adam, Adagrad, ASGD, SimpleSGD).
* `--loss` (default='NLLLoss', type=str): type of loss function to use (NLLLoss, CrossEntropyLoss, MarginLoss, SpreadLoss).
* `--emsize-word` (default=300, type=int): size of word embeddings. Standard values usually don't need to be optimized (BERT 768, Word2Vec 300).
* `--emsize-posi` (default=5, type=int):size of the position embeddings.
* `--maxlength` (default=-1, type=int): maximum length of a sequence (use -1 for determining the length from the training data). Limit max length if most sentences shorter than max lenght.
* `--context-window` (default=0, type=int): size of the moving window of left and right contexts for concatenatation. Context window looks on neigboring words and concatenates them. This helps if a lot of data is avaialable.
* `--convfilters` (default='1024,1024,1024', type=str): number of convolution filters to apply.
* `--convwindows` (default='3,4,5', type=str): sizes of the moving convolutional window.
* `--convstrides` (default='1,1,1', type=str): strides of convolutions.
* `--conv-activation` (default='ReLU', type=str): activation function to use after convolutinal layer (ReLU, GeLU, Tanh).
* `--nhid` (default=200, type=int): number of hidden units.
* `--lr` (default=.1, type=float): initial learning rate.
* `--lr-decay` (default=0.25, type=float): decay amount of learning learning rate if no validation improvement occurs. This depends on optimizer.
* `--l1reg` (default=.0, type=float): add l1 regularuzation loss.
* `--clip` (default=-1, type=float): , gradient clipping (set to -1 to avoid clipping)
* `--batch-size` (default=64, type=int, metavar='N'):  batch size.
* `--dropout` (default=0.5, type=float): dropout applied to layers (0 = no dropout)
* `--seed` (default=1111, type=int): random seed as initialized randomly.
* `--nlines` (default=-1, type=int, metavar='N'): debug param, number of lines to process, -1 for all.
* `--status-reports` (default=3, type=int, metavar='N'): logging param, generate N reports during one training epoch (N=min(N, nbatches)).
* `--init-emword` (default='', type=str): path to initial word embedding; emsize-word must match size of embedding.
* `--fix-emword` (action='store_true'): Specify if the word embedding should be excluded from further optimization.
* `--shuffle-samples` (action='store_true'): shuffle samples.
* `--shuffle-batches` (action='store_true): shuffle batches.
* `--cuda` (action='store_true'): use CUDA if available

### Training

Specify the location of the training directory and the testing directory

`python lttc.py --lang=en --train=data/SMSSpamCollection/train --test=data/SMSSpamCollection./test`

you can also omit the testing directory. The model will then be trained on each of the subdirectories (i.e. the full data, train + test)

`python lttc.py --lang=en --train=data/SMSSpamCollection`

The application will save a preprocessed version of the documents in the dataset folder next to the .txt files (e.g. data/SMSSpamCollection/train/ham/docs1.txt__en_core_web_sm__2.1.0.pkl). If you change the content of the txt file, you'll have to remove the '.pkl' file manually.

The model will train 100 epochs and it will provide you with some status output. If the F-score of the testset improves the model is stored by default in './savedmodels/model', after each epoch you will see the F-scores and at which epoch the best score was reached. You can cancel the process once the scores do not improve anymore (or increase the number of epochs with --epochs=200). If you have a GPU available for training pass the '--cuda' parameter and training will be much faster.

if you want to use a pre-trained embedding use the --init-emword switch, e.g.

`python lttc.py --lang=en --train=data/SMSSpamCollection/train --test=data/SMSSpamCollection/test --init-emword=embedding/wiki.simple.bin`

If using BERT have a look on the [Hugging Face Library](https://github.com/huggingface/pytorch-pretrained-BERT)

## Using / Predicting

specify the directory which contains the model files. The directory should contain these files:

* model.pt
* model.predictions.tsv
* ndx_vocab.txt
* ndx_classes.txt
* ndx_position.txt
* parameters.pkl

You can start a TCP server, serving the model with:

`python lttc.py --serve --model=savedmodels/{modelname}`

By default it uses the host 127.0.0.1 and port 8881. You can specify different by using the --server flag, e.g.:

`python lttc.py --serve --model=savedmodels/SMSSpamCollection_default --server=0.0.0.0:8882`

You can use any TCP client of your choice to test the model. Here is a short example using netcat:

`
echo 'WINNER! Credit for free!' | nc 127.0.0.1 8881
echo "Hey Mom, I'm coming late for dinner." | nc 127.0.0.1 8881
`

The output for each command will be something like:

```bash
spam    -0.4058372974395752     spam:-0.4058 ham:-1.0979
ham     -0.0028526782989501953  spam:-5.8609 ham:-0.0029
````

The fields are tab separated, first field is the identified class, second field is the logarithmic value of the confidence value (class probability), the third field is a summary of log probabilities of all classes. Note the the input format must be UTF-8 encoded and one line per document!

You can also pass an entire file this way, e.g.

`cat data/SMSSpamCollection/test/spam/docs.txt | nc 127.0.0.1 8881`
