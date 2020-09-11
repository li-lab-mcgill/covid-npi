#/usr/bin/python

from __future__ import print_function

import argparse
import torch
import pickle 
import numpy as np 
import os 
import math 
import random 
# import matplotlib.pyplot as plt 
# import seaborn as sns
import scipy.io
import json
import time

import data 

# from sklearn.decomposition import PCA
from torch import nn, optim
from torch.nn import functional as F

from mixmedia import MixMedia
from utils import nearest_neighbors, get_topic_coherence

# from IPython.core.debugger import set_trace

import sys, importlib
importlib.reload(sys.modules['data'])

from torch.utils.tensorboard import SummaryWriter
time_stamp = time.strftime("%m-%d-%H-%M", time.localtime())
print(f"Experiment time stamp: {time_stamp}")

writer = SummaryWriter(f"runs/{time_stamp}")

parser = argparse.ArgumentParser(description='The Embedded Topic Model')

### data and file related arguments
# parser.add_argument('--dataset', type=str, default='GPHIN', help='name of corpus')
# parser.add_argument('--data_path', type=str, default='data/GPHIN', help='directory containing data')

# parser.add_argument('--dataset', type=str, default='WHO_all', help='name of corpus')
# parser.add_argument('--data_path', type=str, default='../../data/WHO_June10/WHO_all', help='directory containing data')

# parser.add_argument('--dataset', type=str, default='GPHIN', help='name of corpus')
# parser.add_argument('--data_path', type=str, default='../../data/GPHIN_labels/gphin_media_data', help='directory containing data')

# parser.add_argument('--dataset', type=str, default='Aylien', help='name of corpus')
# parser.add_argument('--data_path', type=str, default='/Users/yueli/Projects/covid19_media/data/Aylien', help='directory containing data')

parser.add_argument('--dataset', type=str, default='gphin_all_sources', help='name of corpus')
parser.add_argument('--data_path', type=str, default='../../data/GPHIN_labels/gphin_all_sources', help='directory containing data')

# parser.add_argument('--dataset', type=str, default='GPHIN_all', help='name of corpus')
# parser.add_argument('--data_path', type=str, default='/Users/yueli/Projects/covid19_media/pnair6/new_data/GPHIN_all', help='directory containing data')


parser.add_argument('--emb_path', type=str, default='/Users/yueli/Projects/covid19_media/data/trained_word_emb_aylien.txt', help='directory containing embeddings')
# parser.add_argument('--emb_path', type=str, default='/Users/yueli/Projects/covid19_media/data/skipgram_emb_300d.txt', help='directory containing embeddings')

parser.add_argument('--save_path', type=str, default='/Users/yueli/Projects/covid19_media/results/mixmedia', help='path to save results')

parser.add_argument('--batch_size', type=int, default=1000, help='number of documents in a batch for training')

parser.add_argument('--min_df', type=int, default=10, help='to get the right data..minimum document frequency')
# parser.add_argument('--min_df', type=int, default=100, help='to get the right data..minimum document frequency')

### model-related arguments
parser.add_argument('--num_topics', type=int, default=10, help='number of topics')

parser.add_argument('--rho_size', type=int, default=300, help='dimension of rho')
parser.add_argument('--emb_size', type=int, default=300, help='dimension of embeddings')
parser.add_argument('--t_hidden_size', type=int, default=800, help='dimension of hidden space of q(theta)')
parser.add_argument('--theta_act', type=str, default='relu', help='tanh, softplus, relu, rrelu, leakyrelu, elu, selu, glu)')

parser.add_argument('--train_embeddings', type=int, default=1, help='whether to fix rho or train it')

parser.add_argument('--eta_nlayers', type=int, default=3, help='number of layers for eta')
parser.add_argument('--eta_hidden_size', type=int, default=200, help='number of hidden units for rnn')

parser.add_argument('--delta', type=float, default=0.005, help='prior variance')

# q_theta LSTM arguments
parser.add_argument('--one_hot_qtheta_emb', type=int, default=1, help='whther to use 1-hot embedding as q_theta input')
parser.add_argument('--q_theta_arc', type=str, default='lstm', help='q_theta model structure (lstm, trm or electra)', choices=['lstm', 'trm', 'electra'])
parser.add_argument('--q_theta_layers', type=int, default=1, help='number of layers for q_theta')
parser.add_argument('--q_theta_hidden_size', type=int, default=256, help='number of hidden units for q_theta')
parser.add_argument('--q_theta_heads', type=int, default=4, help='number of attention heads for q_theta')
parser.add_argument('--q_theta_drop', type=float, default=0.1, help='dropout rate for q_theta')
parser.add_argument('--q_theta_bi', type=int, default=1, help='whether to use bidirectional LSTM for q_theta')

# country npi LSTM arguments
parser.add_argument('--cnpi_hidden_size', type=int, default=64, help='country npi lstm hidden size')
parser.add_argument('--cnpi_drop', type=float, default=0.1, help='dropout rate for country npi lstm')
parser.add_argument('--cnpi_layers', type=int, default=1, help='number of layers for country npi lstm')

### optimization-related arguments
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--lr_factor', type=float, default=4.0, help='divide learning rate by this')

parser.add_argument('--epochs', type=int, default=150, help='number of epochs to train')

parser.add_argument('--mode', type=str, default='train', help='train or eval model')
# parser.add_argument('--mode', type=str, default='eval_model', help='train or eval model')

parser.add_argument('--optimizer', type=str, default='adam', help='choice of optimizer')
parser.add_argument('--seed', type=int, default=2020, help='random seed (default: 1)')

parser.add_argument('--enc_drop', type=float, default=0.1, help='dropout rate on encoder')
parser.add_argument('--eta_dropout', type=float, default=0.1, help='dropout rate on rnn for eta')

parser.add_argument('--clip', type=float, default=2.0, help='gradient clipping')
parser.add_argument('--nonmono', type=int, default=10, help='number of bad hits allowed')
parser.add_argument('--wdecay', type=float, default=1.2e-6, help='some l2 regularization')

parser.add_argument('--anneal_lr', type=int, default=1, help='whether to anneal the learning rate or not')
parser.add_argument('--bow_norm', type=int, default=1, help='normalize the bows or not')

### evaluation, visualization, and logging-related arguments
parser.add_argument('--num_words', type=int, default=20, help='number of words for topic viz')
parser.add_argument('--log_interval', type=int, default=10, help='when to log training')

parser.add_argument('--visualize_every', type=int, default=1, help='when to visualize results')
parser.add_argument('--eval_batch_size', type=int, default=1000, help='input batch size for evaluation')
parser.add_argument('--load_from', type=str, default='', help='the name of the ckpt to eval from')
parser.add_argument('--tc', type=int, default=0, help='whether to compute tc or not')

parser.add_argument('--predict_labels', type=int, default=1, help='whether to predict labels')
parser.add_argument('--multiclass_labels', type=int, default=1, help='whether to predict labels')

parser.add_argument('--predict_cnpi', type=int, default=1, help='whether to predict country level npi')

parser.add_argument('--time_prior', type=int, default=1, help='whether to use time-dependent topic prior')
parser.add_argument('--source_prior', type=int, default=1, help='whether to use source-specific topic prior')


args = parser.parse_args()


# pca seems unused
# pca = PCA(n_components=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## set seed
np.random.seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.manual_seed(args.seed)

## get data
# 1. vocabulary
print('Getting vocabulary ...')
data_file = os.path.join(args.data_path, 'min_df_{}'.format(args.min_df))

vocab, train, valid, test, q_theta_input_dim = data.get_data(data_file, temporal=True, predict=args.predict_labels, \
    use_time=args.time_prior, use_source=args.source_prior, if_one_hot=args.one_hot_qtheta_emb, q_theta_arc=args.q_theta_arc)

vocab_size = len(vocab)
args.vocab_size = vocab_size


# 1. training data
print('Getting training data ...')
train_tokens = train['tokens']
train_counts = train['counts']
train_embs = train['embs']
train_times = train['times']
train_sources = train['sources']
train_labels = train['labels']

# sort training samples in order of length
train_lengths = [len(train_emb) for train_emb in train_embs]
train_indices_order = np.argsort(train_lengths)

# args.q_theta_input_dim = train_embs[0].shape[1]
args.q_theta_input_dim = q_theta_input_dim

if len(train_labels.shape) == 2 and args.multiclass_labels == 0:
    print("multiclass_labels is turned off but multi-class label file is provided")
    print("multiclass_labels has been turned on.")
    args.multiclass_labels = 1


# args.num_times = len(np.unique(train_times))
if args.time_prior:
    timestamps_file = os.path.join(data_file, 'timestamps.pkl')
    all_timestamps = pickle.load(open(timestamps_file, 'rb'))
    args.num_times = len(all_timestamps)
else:
    args.num_times = 1

args.num_docs_train = len(train_tokens)

# get all sources
if args.source_prior:
    sources_map_file = os.path.join(data_file, 'sources_map.pkl')
    sources_map = pickle.load(open(sources_map_file, 'rb'))
    args.num_sources = len(sources_map)
else:
    args.num_sources = 1


# get all labels
if args.predict_labels:
    labels_map_file = os.path.join(data_file, 'labels_map.pkl')
    labels_map = pickle.load(open(labels_map_file, 'rb'))
    args.num_labels = len(labels_map)
else:
    args.num_labels = 0

# get cnpis
if args.predict_cnpi:
    # load cnpis
    cnpis_file = os.path.join(data_file, 'cnpis.pkl')
    with open(cnpis_file, 'rb') as file:
        cnpis = pickle.load(file)
    args.num_cnpis = cnpis.shape[-1]
    cnpis = torch.from_numpy(cnpis).to(device)
    # load mask
    cnpi_mask_file = os.path.join(data_file, 'cnpi_mask.pkl')
    with open(cnpi_mask_file, 'rb') as file:
        cnpi_mask = pickle.load(file)
    cnpi_mask = torch.from_numpy(cnpi_mask).type('torch.LongTensor').to(device)
    cnpi_mask = cnpi_mask.unsqueeze(-1).expand(cnpis.size())    # match cnpis' shape to apply masking

else:
    cnpis = None
    cnpi_mask = None
    args.num_cnpis = None

train_rnn_inp = data.get_rnn_input(
    train_tokens, train_counts, train_times, train_sources, train_labels,
    args.num_times, args.num_sources, 
    args.vocab_size, args.num_docs_train)


# 2. dev set
print('Getting validation data ...')
valid_tokens = valid['tokens']
valid_counts = valid['counts']
valid_embs = valid['embs']
valid_times = valid['times']
valid_sources = valid['sources']
valid_labels = train['labels']

# sort valid samples in order of length
valid_lengths = [len(valid_emb) for valid_emb in valid_embs]
valid_indices_order = np.argsort(valid_lengths)


args.num_docs_valid = len(valid_tokens)
valid_rnn_inp = data.get_rnn_input(
    valid_tokens, valid_counts, valid_times, valid_sources, valid_labels,
    args.num_times, args.num_sources, 
    args.vocab_size, args.num_docs_valid)

# 3. test data
print('Getting testing data ...')
test_tokens = test['tokens']
test_counts = test['counts']
test_embs = test['embs']
test_times = test['times']
test_sources = test['sources']
test_labels = test['labels']

# sort test samples in order of length
test_lengths = [len(test_emb) for test_emb in test_embs]
test_indices_order = np.argsort(test_lengths)


args.num_docs_test = len(test_tokens)
test_rnn_inp = data.get_rnn_input(
    test_tokens, test_counts, test_times, test_sources, test_labels,
    args.num_times, args.num_sources, 
    args.vocab_size, args.num_docs_test)


test_1_tokens = test['tokens_1']
test_1_counts = test['counts_1']
test_1_embs = test['embs_1']
test_1_times = test_times
args.num_docs_test_1 = len(test_1_tokens)
test_1_rnn_inp = data.get_rnn_input(
    test_1_tokens, test_1_counts, test_1_times, test_sources, test_labels,
    args.num_times, args.num_sources, 
    args.vocab_size, args.num_docs_test)


test_2_tokens = test['tokens_2']
test_2_counts = test['counts_2']
test_2_embs = test['embs_2']
test_2_times = test_times
args.num_docs_test_2 = len(test_2_tokens)
test_2_rnn_inp = data.get_rnn_input(
    test_2_tokens, test_2_counts, test_2_times, test_sources, test_labels,
    args.num_times, args.num_sources, 
    args.vocab_size, args.num_docs_test)


## get word embeddings 
print('Getting word embeddings ...')
emb_path = args.emb_path
vect_path = os.path.join(args.data_path.split('/')[0], 'embeddings.pkl')   
vectors = {}
with open(emb_path, 'rb') as f:
    for l in f:
        line = l.decode().split()
        word = line[0]
        if word in vocab:
            vect = np.array(line[1:]).astype(np.float)
            vectors[word] = vect
word_embeddings = np.zeros((vocab_size, args.emb_size))
words_found = 0
for i, word in enumerate(vocab):
    try: 
        word_embeddings[i] = vectors[word]
        words_found += 1
    except KeyError:
        word_embeddings[i] = np.random.normal(scale=0.6, size=(args.emb_size, ))
word_embeddings = torch.from_numpy(word_embeddings)
args.embeddings_dim = word_embeddings.size()


print('\n')
print('=*'*100)
print('Training a MixMedia Model on {} with the following settings: {}'.format(args.dataset.upper(), args))
print('=*'*100)

if args.mode == 'eval':
    ckpt = args.load_from
else:
    ckpt = os.path.join(args.save_path, time_stamp)

if not os.path.exists(ckpt):
    os.makedirs(ckpt)

## define model and optimizer
if args.load_from != '':
    print('Loading checkpoint from {}'.format(args.load_from))
    with open(args.load_from, 'rb') as f:
        model = torch.load(f)
else:
    model = MixMedia(args, word_embeddings)
print('\nMS-DETM architecture: {}'.format(model))
model.to(device)


if args.optimizer == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
elif args.optimizer == 'adagrad':
    optimizer = optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
elif args.optimizer == 'adadelta':
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
elif args.optimizer == 'rmsprop':
    optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
elif args.optimizer == 'asgd':
    optimizer = optim.ASGD(model.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)
else:
    print('Defaulting to vanilla SGD')
    optimizer = optim.SGD(model.parameters(), lr=args.lr)


def train(epoch):
    """
        Train S-DETM on data for one epoch.
    """
    model.train()
    acc_loss = 0
    acc_nll = 0
    acc_kl_theta_loss = 0
    acc_kl_eta_loss = 0
    acc_kl_alpha_loss = 0
    acc_pred_loss = 0 # classification loss
    acc_cnpi_pred_loss = 0
    cnt = 0

    # indices = torch.randperm(args.num_docs_train)
    indices = torch.tensor(train_indices_order)
    indices = torch.split(indices, args.batch_size)
    
    for idx, ind in enumerate(indices):

        optimizer.zero_grad()
        model.zero_grad()        
        
        data_batch, embs_batch, times_batch, sources_batch, labels_batch = data.get_batch(
            train_tokens, train_counts, train_embs, ind, train_sources, train_labels, 
            args.vocab_size, args.emb_size, temporal=True, times=train_times, if_one_hot=args.one_hot_qtheta_emb, emb_vocab_size=q_theta_input_dim)        

        sums = data_batch.sum(1).unsqueeze(1)

        if args.bow_norm:
            normalized_data_batch = data_batch / sums
        else:
            normalized_data_batch = data_batch

        # print("forward passing ...")

        loss, nll, kl_alpha, kl_eta, kl_theta, pred_loss, cnpi_pred_loss = model(data_batch, normalized_data_batch, embs_batch,
            times_batch, sources_batch, labels_batch, cnpis, cnpi_mask, train_rnn_inp, args.num_docs_train)

        # set_trace()

        # print("forward done.")
        # print("backward passing ...")

        # set_trace()

        loss.backward()

        # print("backward done.")

        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        # set_trace()

        acc_loss += torch.sum(loss).item()
        acc_nll += torch.sum(nll).item()
        acc_kl_theta_loss += torch.sum(kl_theta).item()
        acc_kl_eta_loss += torch.sum(kl_eta).item()
        acc_kl_alpha_loss += torch.sum(kl_alpha).item()

        acc_pred_loss += torch.sum(pred_loss).item()
        acc_cnpi_pred_loss += torch.sum(cnpi_pred_loss).item()

        cnt += 1

        cur_loss = round(acc_loss / cnt, 2) 
        cur_nll = round(acc_nll / cnt, 2) 
        cur_kl_theta = round(acc_kl_theta_loss / cnt, 2) 
        cur_kl_eta = round(acc_kl_eta_loss / cnt, 2) 
        cur_kl_alpha = round(acc_kl_alpha_loss / cnt, 2)

        cur_pred_loss = round(acc_pred_loss / cnt, 2) 
        cur_cnpi_pred_loss = round(acc_cnpi_pred_loss / cnt, 2) 

        if idx % args.log_interval == 0 and idx > 0:

            lr = optimizer.param_groups[0]['lr']
            print('Epoch: {} .. batch: {}/{} .. LR: {} .. KL_theta: {} .. KL_eta: {} .. KL_alpha: {} .. Rec_loss: {} .. Pred_loss: {} .. CNPI_loss: {} .. NELBO: {}'.format(
                epoch, idx, len(indices), lr, cur_kl_theta, cur_kl_eta, cur_kl_alpha, cur_nll, cur_pred_loss, cur_cnpi_pred_loss, cur_loss))

    # tensorboard stuff
    writer.add_scalar('LR', lr, epoch)
    writer.add_scalar('KL_theta', cur_kl_theta, epoch)
    writer.add_scalar('KL_eta', cur_kl_eta, epoch)
    writer.add_scalar('KL_alpha', cur_kl_alpha, epoch)
    writer.add_scalar('Rec_loss', cur_nll, epoch)
    writer.add_scalar('Pred_loss', cur_pred_loss, epoch)
    writer.add_scalar('CNPI_loss', cur_cnpi_pred_loss, epoch)
    writer.add_scalar('NELBO', cur_loss, epoch)
    
    lr = optimizer.param_groups[0]['lr']
    print('*'*100)
    print('Epoch----->{} .. LR: {} .. KL_theta: {} .. KL_eta: {} .. KL_alpha: {} .. Rec_loss: {} .. Pred_loss: {} .. CNPI_loss: {} .. NELBO: {}'.format(
            epoch, lr, cur_kl_theta, cur_kl_eta, cur_kl_alpha, cur_nll, cur_pred_loss, cur_cnpi_pred_loss, cur_loss))
    print('*'*100)


def visualize():
    """Visualizes topics and embeddings and word usage evolution.
    """
    model.eval()
    with torch.no_grad():
        alpha = model.mu_q_alpha
        beta = model.get_beta(alpha) 
        
        print('\n')
        print('#'*100)
        print('Visualize topics...')        
        
        topics_words = []
        for k in range(args.num_topics):
            gamma = beta[k, :]
            top_words = list(gamma.cpu().numpy().argsort()[-args.num_words+1:][::-1])                
            topic_words = [vocab[a] for a in top_words]
            topics_words.append(' '.join(topic_words))
            print('Topic {} .. ===> {}'.format(k, topic_words)) 

        print('\n')
        print('Visualize word embeddings ...')
        # queries = ['economic', 'assembly', 'security', 'management', 'debt', 'rights',  'africa']
        # queries = ['economic', 'assembly', 'security', 'management', 'rights',  'africa']
        queries = ['border', 'vaccines', 'coronaviruses', 'masks']
        queries = set(queries).intersection(vocab)
        try:
            embeddings = model.rho.weight  # Vocab_size x E
        except:
            embeddings = model.rho         # Vocab_size x E
        # neighbors = []
        for word in queries:
            print('word: {} .. neighbors: {}'.format(
                word, nearest_neighbors(word, embeddings, vocab, args.num_words)))
        print('#'*100)



def _eta_helper(rnn_inp):

    etas = torch.zeros(model.num_sources, model.num_times, model.num_topics).to(device)
    inp = model.q_eta_map(rnn_inp.view(rnn_inp.size(0)*rnn_inp.size(1), -1)).view(rnn_inp.size(0),rnn_inp.size(1),-1)
    hidden = model.init_hidden()
    output, _ = model.q_eta(inp, hidden)
    inp_0 = torch.cat([output[:,0,:], torch.zeros(model.num_sources, model.num_topics).to(device)], dim=1)
    etas[:, 0, :] = model.mu_q_eta(inp_0)

    for t in range(1, model.num_times):
        inp_t = torch.cat([output[:,t,:], etas[:, t-1, :]], dim=1)
        etas[:, t, :] = model.mu_q_eta(inp_t)
    
    return etas

def get_eta(data_type):
    model.eval()
    with torch.no_grad():
        if data_type == 'val':
            rnn_inp = valid_rnn_inp
            return _eta_helper(rnn_inp)
        elif data_type == 'test':
            rnn_1_inp = test_1_rnn_inp
            return _eta_helper(rnn_1_inp)
        elif data_type == 'train':
            return _eta_helper(train_rnn_inp)
        else:
            raise Exception('invalid data_type: '.data_type)


def get_theta(eta, embs, times, sources):
    model.eval()
    with torch.no_grad():
        eta_std = eta[sources.type('torch.LongTensor'), times.type('torch.LongTensor')] # D x K
        inp = torch.cat([embs, eta_std], dim=1)
        q_theta = model.q_theta(inp)
        # if args.one_hot_qtheta_emb and args.q_theta_arc != 'electra':
        #     embs = model.q_theta_emb(embs)
        # if model.q_theta_arc == 'trm':
        #     embs = model.pos_encode(embs)
        #     q_theta_out = model.q_theta(embs)        
        # else:
        #     q_theta_out = model.q_theta(embs)[0]
        # # q_theta_out = model.q_theta_att(key=q_theta_out, query=model.q_theta_att_query, value=q_theta_out)[1].squeeze()
        # # q_theta = model.q_theta_att(key=q_theta_out, query=eta_std.unsqueeze(1), value=q_theta_out)[1].squeeze()
        # q_theta_out = torch.max(q_theta_out, dim=1)[0]
        # q_theta = torch.cat([q_theta_out, eta_std], dim=1)
        mu_theta = model.mu_q_theta(q_theta)
        theta = F.softmax(mu_theta, dim=-1)   
        # print(q_theta)   
        return theta

def get_completion_ppl(source):
    """Returns document completion perplexity.
    """
    model.eval()
    with torch.no_grad():
        alpha = model.mu_q_alpha # KxTxL
        if source == 'val':
            indices = torch.tensor(valid_indices_order)
            indices = torch.split(indices, args.eval_batch_size)            
            # indices = torch.split(torch.tensor(range(args.num_docs_valid)), args.eval_batch_size)            
            tokens = valid_tokens
            counts = valid_counts
            embs = valid_embs
            times = valid_times
            sources = valid_sources
            labels = valid_labels

            eta = get_eta('val')

            acc_loss = 0
            acc_pred_loss = 0

            cnt = 0
            for idx, ind in enumerate(indices):
                
                data_batch, embs_batch, times_batch, sources_batch, labels_batch = data.get_batch(
                    tokens, counts, embs, ind, sources, labels, 
                    args.vocab_size, args.emb_size, temporal=True, times=times, if_one_hot=args.one_hot_qtheta_emb, emb_vocab_size=q_theta_input_dim)

                sums = data_batch.sum(1).unsqueeze(1)

                if args.bow_norm:
                    normalized_data_batch = data_batch / sums
                else:
                    normalized_data_batch = data_batch
                
                # theta = get_theta(eta, embs_batch, times_batch, sources_batch)
                theta = get_theta(eta, normalized_data_batch, times_batch, sources_batch)
                                
                beta = model.get_beta(alpha)
                nll = -torch.log(torch.mm(theta, beta)) * data_batch
                nll = nll.sum(-1)
                loss = nll / sums.squeeze()
                loss = loss.mean().item()
                acc_loss += loss                
                
                if args.predict_labels:
                    pred_loss = model.get_prediction_loss(theta, labels_batch)
                    acc_pred_loss += pred_loss / data_batch.size(0)

                cnt += 1

            cur_loss = acc_loss / cnt            

            ppl_all = round(math.exp(cur_loss), 1)

            if args.predict_labels:
                cur_pred_loss = acc_pred_loss / cnt
                pdl_all = round(cur_pred_loss.item(), 2)
            else:
                pdl_all = 0

            print('*'*100)
            print('{} PPL: {} .. PDL: {}'.format(source.upper(), ppl_all, pdl_all))
            print('*'*100)
            return ppl_all, pdl_all
        else: 
            indices = torch.tensor(test_indices_order)
            indices = torch.split(indices, args.eval_batch_size)
            tokens_1 = test_1_tokens
            counts_1 = test_1_counts
            embs_1 = test_1_embs
            tokens_2 = test_2_tokens
            counts_2 = test_2_counts            
            embs_2 = test_2_embs

            eta_1 = get_eta('test')

            acc_loss = 0
            acc_pred_loss = 0

            cnt = 0
            # indices = torch.split(torch.tensor(range(args.num_docs_test)), args.eval_batch_size)
            for idx, ind in enumerate(indices):

                data_batch_1, embs_batch_1, times_batch_1, sources_batch_1, labels_batch_1 = data.get_batch(
                    tokens_1, counts_1, embs_1, ind, test_sources, test_labels,
                    args.vocab_size, args.emb_size, temporal=True, times=test_times, if_one_hot=args.one_hot_qtheta_emb, emb_vocab_size=q_theta_input_dim)
                
                sums_1 = data_batch_1.sum(1).unsqueeze(1)

                if args.bow_norm:
                    normalized_data_batch_1 = data_batch_1 / sums_1
                else:
                    normalized_data_batch_1 = data_batch_1
                
                theta = get_theta(eta_1, normalized_data_batch_1, times_batch_1, sources_batch_1)

                data_batch_2, embs_batch_2, times_batch_2, sources_batch_2, labels_batch_2 = data.get_batch(
                    tokens_2, counts_2, embs_2, ind, test_sources, test_labels,
                    args.vocab_size, args.emb_size, temporal=True, times=test_times, if_one_hot=args.one_hot_qtheta_emb, emb_vocab_size=q_theta_input_dim)

                sums_2 = data_batch_2.sum(1).unsqueeze(1)
                
                beta = model.get_beta(alpha)
                loglik = torch.log(torch.mm(theta, beta))                 
                nll = -loglik * data_batch_2
                nll = nll.sum(-1)
                loss = nll / sums_2.squeeze()
                loss = loss.mean().item()
                acc_loss += loss

                pred_loss = torch.tensor(0)

                if args.predict_labels:
                    pred_loss = model.get_prediction_loss(theta, labels_batch_2)
                    acc_pred_loss += pred_loss / data_batch_1.size(0)

                cnt += 1

            cur_loss = acc_loss / cnt                        
            ppl_dc = round(math.exp(cur_loss), 1)               

            if args.predict_labels:
                cur_pred_loss = acc_pred_loss / cnt
                pdl_dc = round(cur_pred_loss.item(), 2)
            else:
                pdl_dc = 0            

            print('*'*100)
            print('{} Doc Completion PPL: {} .. Doc Classification PDL: {}'.format(source.upper(), ppl_dc, pdl_dc))
            print('*'*100)
            return ppl_dc, pdl_dc

def _diversity_helper(beta, num_tops):
    list_w = np.zeros((args.num_topics, num_tops))
    for k in range(args.num_topics):
        gamma = beta[k, :]
        top_words = gamma.detach().cpu().numpy().argsort()[-num_tops:][::-1]
        list_w[k, :] = top_words
    list_w = np.reshape(list_w, (-1))
    list_w = list(list_w)
    n_unique = len(np.unique(list_w))
    diversity = n_unique / (args.num_topics * num_tops)
    return diversity

def get_topic_quality():
    """Returns topic coherence and topic diversity.
    """
    model.eval()    
    with torch.no_grad():
        alpha = model.mu_q_alpha
        beta = model.get_beta(alpha) 
        print('beta: ', beta.size())

        print('\n')
        print('#'*100)
        print('Get topic diversity...')
        num_tops = 25

        TD_all = _diversity_helper(beta, num_tops)            
        
        TD = np.mean(TD_all)
        print('Topic Diversity is: {}'.format(TD))

        print('\n')
        print('Get topic coherence...')
        print('train_tokens: ', train_tokens[0])
             
        TC_all, cnt_all = get_topic_coherence(beta.cpu().detach().numpy(), train_tokens, vocab)

        TC_all = torch.tensor(TC_all)
        cnt_all = torch.tensor(cnt_all)
        TC_all = TC_all / cnt_all
        TC_all[TC_all<0] = 0

        TC = TC_all.mean().item()
        print('Topic Coherence is: ', TC)
        print('\n')

        print('Get topic quality...')
        TQ = TC * TD
        print('Topic Quality is: {}'.format(TQ))
        print('#'*100)

        return TQ, TC, TD

def compute_top_k_recall(labels, predictions, k=5):
    '''
    inputs:
    - labels: tensor, (number of samples, number of classes)
    - predictions: tensor, (number of samples, number of classes)
    output:
    - top-k recall of the batch
    '''
    # remove ones without positive labels
    has_pos_labels = labels.sum(1) != 0
    labels = labels[has_pos_labels, :]
    predictions = predictions[has_pos_labels, :]
    idxs = torch.argsort(predictions, dim=1, descending=True)[:, 0: k]
    return (torch.gather(labels, 1, idxs).sum(1) / labels.sum(1)).mean().item()

def get_cnpi_top_k_recall(cnpis, cnpi_mask, mode):
    assert mode in ['val', 'test'], 'mode must be val or test'

    with torch.no_grad():
        eta = get_eta(mode)
        predictions = model.cnpi_lstm(eta)[0]
        predictions = model.cnpi_out(predictions)
        cnpi_mask = 1 - cnpi_mask   # invert the mask to use unseen data points for evaluation
        cnpis_masked = cnpis * cnpi_mask
        predictions_masked = predictions * cnpi_mask    # taking indices only so not computing sigmoid
    return {
        1: [compute_top_k_recall(cnpis_masked.reshape(-1, cnpis_masked.shape[-1]), \
            predictions_masked.reshape(-1, predictions_masked.shape[-1]), 1)],
        3: [compute_top_k_recall(cnpis_masked.reshape(-1, cnpis_masked.shape[-1]), \
            predictions_masked.reshape(-1, predictions_masked.shape[-1]), 3)],
        5: [compute_top_k_recall(cnpis_masked.reshape(-1, cnpis_masked.shape[-1]), \
            predictions_masked.reshape(-1, predictions_masked.shape[-1]), 5)],
        10: [compute_top_k_recall(cnpis_masked.reshape(-1, cnpis_masked.shape[-1]), \
            predictions_masked.reshape(-1, predictions_masked.shape[-1]), 10)],
        }

if args.mode == 'train':
    ## train model on data by looping through multiple epochs
    best_epoch = 0
    best_val_ppl = 1e9
    all_val_ppls = []
    all_val_pdls = []

    if args.anneal_lr:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.25, patience=10, min_lr=1e-7)
    
    for epoch in range(1, args.epochs):
        train(epoch)
        
        if epoch % args.visualize_every == 0:
            visualize()
        
        # print(model.classifier.weight)

        val_ppl, val_pdl = get_completion_ppl('val')

        # tensorboard stuff
        writer.add_scalar('PPL/val', val_ppl, epoch)
        if (epoch - 1) % 5 == 0:
            tq, tc, td = get_topic_quality()
            writer.add_scalar('Topic/quality', tq, epoch)
            writer.add_scalar('Topic/coherence', tc, epoch)
            writer.add_scalar('Topic/diversity', td, epoch)
        if args.predict_cnpi:
            # cnpi top k recall on validation set
            val_cnpi_top_ks = get_cnpi_top_k_recall(cnpis, cnpi_mask, 'val')
            for k, recall in val_cnpi_top_ks.items():
                writer.add_scalar(f"Val_top_k_recall/{k}", recall[0], epoch)
        
        if val_ppl < best_val_ppl:
            with open(os.path.join(ckpt, 'model.pt'), 'wb') as f:
                torch.save(model, f) # UNCOMMENT FOR REAL RUN
            best_epoch = epoch
            best_val_ppl = val_ppl
        else:
            ## check whether to anneal lr
            # lr = optimizer.param_groups[0]['lr']
            # if args.anneal_lr and (len(all_val_ppls) > args.nonmono and val_ppl > np.mean(all_val_ppls[:-args.nonmono]) and lr > 1e-5):
            #     optimizer.param_groups[0]['lr'] /= args.lr_factor
            if args.anneal_lr:
                scheduler.step(val_ppl)
        all_val_ppls.append(val_ppl)
        all_val_pdls.append(val_pdl)

    # with open(ckpt, 'rb') as f:
    #     model = torch.load(f)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
                
        print('saving topic matrix beta...')
        alpha = model.mu_q_alpha
        beta = model.get_beta(alpha).cpu().detach().numpy()
        np.save(os.path.join(ckpt, 'beta.npy'), beta, allow_pickle=False)

        
        print('saving alpha...')
        alpha = model.mu_q_alpha.cpu().detach().numpy()
        np.save(os.path.join(ckpt, 'alpha.npy'), alpha, allow_pickle=False)


        print('saving classifer weights...')
        classifer_weights = model.classifier.weight.cpu().detach().numpy()
        np.save(os.path.join(ckpt, 'classifer.npy'), classifer_weights, allow_pickle=False)

        print('saving eta ...')
        eta = get_eta('train').cpu().detach().numpy()
        np.save(os.path.join(ckpt, 'eta.npy'), eta, allow_pickle=False)

        if args.train_embeddings:
            print('saving word embedding matrix rho...')
            rho = model.rho.weight.cpu().detach().numpy()
            np.save(os.path.join(ckpt, 'rho.npy'), rho, allow_pickle=False)

        f=open(os.path.join(ckpt, 'val_ppl.txt'),'w')
        s1='\n'.join([str(i) for i in all_val_ppls])        
        f.write(s1)
        f.close()

        f=open(os.path.join(ckpt, 'val_pdl.txt'),'w')
        s1='\n'.join([str(i) for i in all_val_pdls])        
        f.write(s1)
        f.close()
else: 
    with open(ckpt, 'rb') as f:
        model = torch.load(f)
    model = model.to(device)

# dumping configurations to disk
config_dict = {
    'model_type': 'mixmedia',
    'dataset': args.dataset,
    'K': args.num_topics,
    'theta_hidden_size': args.t_hidden_size,
    'clipping': args.clip,
    'lr': args.lr,
    'batch_size': args.batch_size,
    'rho_size': args.rho_size,
    'eta_nlayers': args.eta_nlayers,
    'min_df': args.min_df,
    'train_embeddings': args.train_embeddings,
    'predict_labels': args.predict_labels,
    'time_prior': args.time_prior,
    'source_prior': args.source_prior,
}

if args.predict_cnpi:
    config_dict['cnpi_hidden_size'] = args.cnpi_hidden_size
    config_dict['cnpi_drop'] = args.cnpi_drop
    config_dict['cnpi_layers'] = args.cnpi_layers

with open(os.path.join(ckpt, 'config.json'), 'w') as file:
    json.dump(config_dict, file)

print('computing validation perplexity...')
val_ppl, val_pdl = get_completion_ppl('val')

if args.predict_cnpi:
    # cnpi top k recall on validation set
    val_cnpi_top_ks = get_cnpi_top_k_recall(cnpis, cnpi_mask, 'val')
    print('\ntop-k recalls on val:')
    for k, recall in val_cnpi_top_ks.items():
        print(f'top-{k}: {recall}')
    with open(os.path.join(ckpt, 'val_cnpi_top_ks.json'), 'w') as file:
        json.dump(val_cnpi_top_ks, file)

print('computing test perplexity...')
test_ppl, test_pdl = get_completion_ppl('test')

if args.predict_cnpi:
    # cnpi top k recall on test set
    test_cnpi_top_ks = get_cnpi_top_k_recall(cnpis, cnpi_mask, 'test')
    print('\ntop-k recalls on test:')
    for k, recall in test_cnpi_top_ks.items():
        print(f'top-{k}: {recall}')
    with open(os.path.join(ckpt, 'test_cnpi_top_ks.json'), 'w') as file:
        json.dump(test_cnpi_top_ks, file)

f=open(os.path.join(ckpt, 'test_ppl.txt'),'w')
f.write(str(test_ppl))
f.close()

f=open(os.path.join(ckpt, 'test_pdl.txt'),'w')
f.write(str(test_pdl))
f.close()    

tq, tc, td = get_topic_quality()

f=open(os.path.join(ckpt, 'tq.txt'),'w')
s1="Topic Quality: "+str(tq)
s2="Topic Coherence: "+str(tc)
s3="Topic Diversity: "+str(td)
f.write(s1+'\n'+s2+'\n'+s3+'\n')
f.close()

f=open(os.path.join(ckpt, 'tq.txt'),'r')
[print(i,end='') for i in f.readlines()]
f.close()

print('visualizing topics and embeddings...')
visualize()









