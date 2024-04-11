import sys
import os
import argparse
import shutil
import math
import numpy as np
import pandas as pd
import scipy.sparse as ss
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from datetime import datetime
import time
import configparser
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
import gc
import importlib.machinery
import importlib.util
import lib.Metrics
import lib.Utils
from model.model import Model

################# python input parameters #######################
parser = argparse.ArgumentParser()
parser.add_argument('-model', type=str, default='staeformer', help='choose which model to train and test')
parser.add_argument('-version', type=str, default=0, help='train version')
parser.add_argument('-note', type=str, default='', help='additional information')
parser.add_argument('-instep', type=int, default=12, help='input step')
parser.add_argument('-outstep', type=int, default=12, help='predict step')
parser.add_argument('-hc', type=int, default=32, help='hidden channel')
parser.add_argument('-batch', type=int, default=16, help='batch size')
parser.add_argument('-epoch', type=int, default=500, help='training epochs')
parser.add_argument('-gs', type=int, default=1, help='mean,std')
parser.add_argument('-addtime', default=False, action="store_true", help='Add timestamp')
parser.add_argument('-adj', default=False, action="store_true", help='Add adj')
parser.add_argument('-mode', type=str, default='train', help='train or eval0')
parser.add_argument("-debug", "-de", default=False, action="store_true")
parser.add_argument('-data', type=str, default='pems04', help='pems03,04,07,08')
parser.add_argument('-train', type=float, default=0.6, help='train data: 0.8,0.7,0.6,0.5')
parser.add_argument('-scaler', type=str, default='zscore', help='data scaler process type, zscore or minmax')
parser.add_argument('-cuda', type=int, default=3, help='cuda device number')
parser.add_argument('-loss', type=str, default='quantile', help='loss function, combine1, combine2, combine3')
parser.add_argument('-seed', default=None, help='torch & numpy seed')
parser.add_argument('-qcl', default=False, action="store_true", help='if use quantile cl')
parser.add_argument('-qcl_times', default=10, type=int, help='quantile cl times')
parser.add_argument('-qcl_len', default=0.1, type=float, help='quantile cl length')
parser.add_argument('-qcldes', default=False, action="store_true",help='quantile rank descending')
parser.add_argument('-qcl_size', type=int, default=300, help='qcl_size')

parser.add_argument('-tcl', default=False, action="store_true", help='if add temporal cl')
parser.add_argument('-tcl_size', type=int, default=16, help='tcl_size')

parser.add_argument('-scl', default=False, action="store_true", help='if add spatial curriculum')
parser.add_argument('-descending','-des', default=False, action="store_true", help='sort descending')
parser.add_argument('-wait_iter','-wscl', default=100, type=int, help='wait how many iters before add spatial curriculum')
parser.add_argument('-scl_size', '-scls', type=int, default=1000, help='quantile spatial cl iter size') 
parser.add_argument('-scl_length', '-scll', type=int, default=10, help='spatial cl length') 

parser.add_argument('-we', default=False, action="store_true", help='if add cl weight in loss calculatation')
parser.add_argument('-clt', type=int, default=100, help='T (epoch) in cl')
parser.add_argument('-retain', "-rt", type=float, default=0.3, help='retain in cl')
parser.add_argument('--rho', type=float, default=0.1, help='the factor of determining the radius')
parser.add_argument('-clweithT', "-clwt", type=int, default=100, help='coverage epoch for vanilla model')

parser.add_argument('-stdrop', '-std', type=int, default=0, help='if add stcdropout')  
parser.add_argument('-sort_rank', '-rk', type=str, default='s', help='t, s, st, ts') 
parser.add_argument('-cl_epoch', '-cle', type=int, default=10, help='quantile rank cl epoch') 
# parser.add_argument('-iter_cl', '-icl', type=int, default=1, help='repeat cl times') 
parser.add_argument('-start_epoch', '-se', type=int, default=0, help='when start quantile rank cl') 
args = parser.parse_args()  # python
# args = parser.parse_args(args=[])    # jupyter notebook
device = torch.device("cuda:{}".format(args.cuda)) if torch.cuda.is_available() else torch.device("cpu")
################# Global Parameters setting #######################
if args.data=='pems04':
    data_path = f"./data/staeformer/PEMS04"
if args.data=='pems03':
    data_path = f"./data/staeformer/PEMS03"    
if args.data=='pems07':
    data_path = f"./data/staeformer/PEMS07" 
if args.data=='pems08':
    data_path = f"./data/staeformer/PEMS08"        
if args.data=='metrla':
    data_path = f"./data/staeformer/METRLA"    
if args.data=='pemsbay':
    data_path = f"./data/staeformer/PEMSBAY"  
IFGS = True if args.gs == 1 else False
DATANAME = args.data
quantiles = [0.1, 0.5, 0.9]
MODELNAME = args.model
BATCHSIZE = args.batch
EPOCH = args.epoch
if args.debug:
    EPOCH = 150
TIMESTEP_IN = args.instep
TIMESTEP_OUT = args.outstep
LOSS = args.loss
NOTE = args.note
SEED = args.seed
################# Statistic Parameters from init_config.ini #######################
ini_config = configparser.ConfigParser()
ini_config.read('./init_config.ini', encoding='UTF-8')
common_config = ini_config['common']
data_config = ini_config[DATANAME]
STGCN_ADJ = str(data_config['STGCN_ADJ'])
DCRNN_ADJ = str(data_config['DCRNN_ADJ'])
N_NODE = int(data_config['N_NODE'])  # 207,325,228
CHANNEL = 2 if args.addtime else int(common_config['CHANNEL'])  # 1
# LEARNING_RATE = float(common_config['LEARNING_RATE'])   # 0.001
LEARNING_RATE = 0.001
# PATIENCE = int(common_config['PATIENCE'])   # 10
PRINT_EPOCH = 1
PATIENCE = 5
OPTIMIZER = str(common_config['OPTIMIZER'])  # Adam
# LOSS = str(common_config['LOSS'])  # MAE
# TRAIN = float(common_config['TRAIN']) # 0.8
TRAIN = args.train
VAL = float(common_config['VAL'])  # 0.1
TEST = float(common_config['TEST'])  # 0.1
################# random seed setting #######################
if SEED is not None:
    SEED = int(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
################# System Parameter Setting #######################DATANAME = 'PEMS0' + args.data
PATH = "./save/{}_{}_in{}_out{}_addtime{}_adj{}_lr{}_hc{}_train{}_val{}_test{}_seed{}_loss{}_version".format(
    DATANAME, args.model, args.instep, args.outstep, args.addtime, args.adj, LEARNING_RATE, args.hc, TRAIN, VAL, TEST,
    SEED, LOSS)
single_version_PATH = PATH + args.version
single_version_PATH += "_note:{}".format(NOTE)

import os

cpu_num = 1
os.environ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)


##################  data preparation   #############################
def vrange(starts, stops):
    """Create ranges of integers for multiple start/stop

    Parameters:
        starts (1-D array_like): starts for each range
        stops (1-D array_like): stops for each range (same shape as starts)
        
        Lengths of each range should be equal.

    Returns:
        numpy.ndarray: 2d array for each range
        
    For example:

        >>> starts = [1, 2, 3, 4]
        >>> stops  = [4, 5, 6, 7]
        >>> vrange(starts, stops)
        array([[1, 2, 3],
               [2, 3, 4],
               [3, 4, 5],
               [4, 5, 6]])

    Ref: https://codereview.stackexchange.com/questions/83018/vectorized-numpy-version-of-arange-with-multiple-start-stop
    """
    stops = np.asarray(stops)
    l = stops - starts  # Lengths of each range. Should be equal, e.g. [12, 12, 12, ...]
    assert l.min() == l.max(), "Lengths of each range should be equal."
    indices = np.repeat(stops - l.cumsum(), l) + np.arange(l.sum())
    return indices.reshape(-1, l[0])

def get_dataloaders_from_index_data(
    data_dir, tod=True, dow=True, dom=False, batch_size=16, log=None
):
    data = np.load(os.path.join(data_dir, "data.npz"))["data"].astype(np.float32)

    features = [0]
    if tod:
        features.append(1)
    if dow:
        features.append(2)
    # if dom:
    #     features.append(3)
    data = data[..., features]

    index = np.load(os.path.join(data_dir, "index.npz"))

    train_index = index["train"]  # (num_samples, 3)
    val_index = index["val"]
    test_index = index["test"]
#     print('val_index  .',val_index)
#     print('test_index  .',test_index)
#     exit()
    x_train_index = vrange(train_index[:, 0], train_index[:, 1])
    y_train_index = vrange(train_index[:, 1], train_index[:, 2])
    x_val_index = vrange(val_index[:, 0], val_index[:, 1])
    y_val_index = vrange(val_index[:, 1], val_index[:, 2])
    x_test_index = vrange(test_index[:, 0], test_index[:, 1])
    y_test_index = vrange(test_index[:, 1], test_index[:, 2])

    x_train = data[x_train_index]
    y_train = data[y_train_index][..., :1]

    x_val = data[x_val_index]
    y_val = data[y_val_index][..., :1]
    x_test = data[x_test_index]
    y_test = data[y_test_index][..., :1]
    if args.debug:
        x_train = x_train[:128]
        y_train = y_train[:128]
        x_val = x_val[:128]
        y_val = y_val[:128]        
        x_test = x_test[:128]
        y_test = y_test[:128]          
#     print('x_train shape is ',x_train.shape)
#     exit()
    scaler = StandardScaler(mean=np.mean(x_train[..., 0], axis=0), std=np.std(x_train[..., 0],axis=0))

    if IFGS:
        train_mean, train_std = np.mean(x_train[..., 0], axis=(0, 1)), np.std(x_train[..., 0],
                                                                                   axis=(0, 1))
    else:
        train_mean, train_std = np.mean(x_train[..., 0], axis=(0, 1, 2)), np.std(x_train[..., 0],
                                                                                        axis=(0, 1, 2))
        
    x_stats = {'mean': train_mean, 'std': train_std}
    x_train[..., 0] = scaler.transform(x_train[..., 0])
    x_val[..., 0] = scaler.transform(x_val[..., 0])
    x_test[..., 0] = scaler.transform(x_test[..., 0])

#     print_log(f"Trainset:\tx-{x_train.shape}\ty-{y_train.shape}", log=log)
#     print_log(f"Valset:  \tx-{x_val.shape}  \ty-{y_val.shape}", log=log)
#     print_log(f"Testset:\tx-{x_test.shape}\ty-{y_test.shape}", log=log)

    trainset = torch.utils.data.TensorDataset(
        torch.FloatTensor(x_train), torch.FloatTensor(y_train)
    )
    valset = torch.utils.data.TensorDataset(
        torch.FloatTensor(x_val), torch.FloatTensor(y_val)
    )
    testset = torch.utils.data.TensorDataset(
        torch.FloatTensor(x_test), torch.FloatTensor(y_test)
    )

    trainset_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True,drop_last=True
    )
    valset_loader = torch.utils.data.DataLoader(
        valset, batch_size=batch_size, shuffle=False
    )
    testset_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False
    )

    return trainset_loader, valset_loader, testset_loader, x_stats


class StandardScaler:
    """
    Standard the input
    https://github.com/nnzhan/Graph-WaveNet/blob/master/util.py
    """

    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def fit_transform(self, data):
        self.mean = data.mean()
        self.std = data.std()

        return (data - self.mean) / self.std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean
    
def data_preprocess(whichdata):
    if whichdata == 'pems04':
        # data = np.load(os.path.join(data_dir, "data.npz"))["data"].astype(np.float32)
        data = np.load('./data/PEMS04/data.npz')['data'].astype(np.float32)  # [samples,nodes]
        index = np.load('./data/PEMS04/index.npz')
        train_index = index["train"]  # (num_samples, 3)
        val_index = index["val"]
        test_index = index["test"]

        x_train_index = vrange(train_index[:, 0], train_index[:, 1])
        y_train_index = vrange(train_index[:, 1], train_index[:, 2])
        x_val_index = vrange(val_index[:, 0], val_index[:, 1])
        y_val_index = vrange(val_index[:, 1], val_index[:, 2])
        x_test_index = vrange(test_index[:, 0], test_index[:, 1])
        y_test_index = vrange(test_index[:, 1], test_index[:, 2])

        x_train = data[x_train_index]
        y_train = data[y_train_index][..., :1]
        x_val = data[x_val_index]
        y_val = data[y_val_index][..., :1]
        x_test = data[x_test_index]
        y_test = data[y_test_index][..., :1]
        print('x train shape is : ', x_train.shape)
        print('x val shape is : ', x_val.shape)
        print('x_test shape is : ', x_test.shape)
        # exit()
    if whichdata == 'metrla':
        data = pd.read_hdf('./data/metr-la/metr-la.h5')
        data.index.freq = data.index.inferred_freq
        time = pd.DataFrame(data.index, columns=['date'])
        data = data.values
    time["hour"] = time["date"].apply(lambda x: x.hour)
    time["year"] = time["date"].apply(lambda x: x.year)
    time["day_of_week"] = time["date"].apply(lambda x: x.dayofweek)
    hours = time["hour"]
    dows = time["day_of_week"]
    cov_X = np.c_[np.asarray(hours), np.asarray(dows)]  # [samples,2]
    cov_X = np.tile(cov_X, [data.shape[1], 1, 1]).transpose(1, 0, 2)  # [samples,nodes,features=2]
    data = np.concatenate([data[:, :, np.newaxis], cov_X], axis=2)  # [samples,nodes,features=3]
    time_stamp = time["date"].values
    return data, time_stamp


def seq(data, train, if_stats=False):
    # input data shape: [samples,nodes]  [nodes,samples]
    trainval_num = int(data.shape[0] * (train + VAL))
    sample_data = data[:trainval_num, :, 0]
    total_num = int(data.shape[0] * (train + VAL + TEST))
    trainval_data, test_data = [], []  # TV : Train and Val
    if if_stats == True:
        if IFGS:
            train_mean, train_std = np.mean(sample_data, axis=0), np.std(sample_data, axis=0)
        else:
            train_mean, train_std = np.mean(sample_data, axis=(0, 1)), np.std(sample_data, axis=(0, 1))
        x_stats = {'mean': train_mean, 'std': train_std}
        print('train_mean shape :', train_mean.shape)
    if if_stats != True:
        x_stats = None
    for i in range(trainval_num - TIMESTEP_OUT - TIMESTEP_IN + 1):
        xy = data[i:i + TIMESTEP_IN + TIMESTEP_OUT, :]
        trainval_data.append(xy)
    trainval_data = np.array(trainval_data)  # output data shape: [total_batch, time, nodes, channel] [Samples,T,N,C]
    train_data = trainval_data[0:int(trainval_data.shape[0] * (train / (train + VAL)))]
    val_data = trainval_data[int(trainval_data.shape[0] * (train / (train + VAL))):]
    # print('train_data shape is : ', train_data.shape)
    #     print('TRAIN_DATA type is : ',type(TRAIN_DATA))

    for i in range(trainval_num - TIMESTEP_IN, total_num - TIMESTEP_OUT - TIMESTEP_IN + 1):
        xy = data[i:i + TIMESTEP_IN + TIMESTEP_OUT, :]
        test_data.append(xy)
    test_data = np.array(test_data)  # output data shape: [total_batch, time, nodes, channel] [Samples,T,N,C]
    # print('test_data shape is : ', test_data.shape)
    seq_data = {'train': train_data, 'val': val_data, 'test': test_data}
    return seq_data, x_stats


def get_inputdata(data, stamp, train, ifaddtime=False):
    print('data load........')
    timestamp = np.tile(stamp, [data.shape[1], 1]).transpose(1, 0)[:, :, np.newaxis]  # [samples,nodes]
    data2 = np.copy(data)
    print('initial data shape is: ', data.shape)
    seq_data, x_stats = seq(data2, train, if_stats=True)  # [samples,N] -> {train,val,test} [B,T,N,C]
    if ifaddtime:
        timestamp = (timestamp - timestamp.astype("datetime64[D]")) / np.timedelta64(1, "D")
        sca_seq_time, _ = seq(timestamp, train, if_stats=False)  # [samples,N] -> {train,val,test} [B,T,N,C]

    for key in seq_data.keys():
        #         print(key ,seq_data[key][:, 0:TIMESTEP_IN, :, 0:1].shape)
        seq_data[key][:, 0:TIMESTEP_IN, :, 0:1] = lib.Utils.z_score(seq_data[key][:, 0:TIMESTEP_IN, :, 0:1],
                                                                    x_stats['mean'], x_stats['std'])
        if ifaddtime:
            seq_data[key] = np.concatenate((seq_data[key], sca_seq_time[key]), axis=-1)  # [B,T,N,C] -> [B,T,N,C+1]
        if args.debug:
            seq_data[key] = seq_data[key][:128]
    return seq_data, x_stats


def torch_data_loader(device, data, data_type, shuffle=True):
    x = torch.Tensor(data[data_type][:, 0:TIMESTEP_IN, :, :]).to(device)  # [B,T=TIMESTEP_IN,N,C]
    y = torch.Tensor(data[data_type][:, TIMESTEP_IN:TIMESTEP_IN + TIMESTEP_OUT, :, 0:1]).to(
        device)  # [B,T=TIMESTEP_OUT,N,C]
    data = torch.utils.data.TensorDataset(x, y)
    data_iter = torch.utils.data.DataLoader(data, BATCHSIZE, shuffle=shuffle)
    return data_iter


def getModel(name, device):
    model_path = './model/' + args.model + '.py'  # AGCRN.py 的路径
    loader = importlib.machinery.SourceFileLoader('baseline_py_file', model_path)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    baseline_py_file = importlib.util.module_from_spec(spec)
    loader.exec_module(baseline_py_file)

    ########## select the baseline model ##########
    ADJTYPE = 'doubletransition' 
    if args.model == 'stgcn17':
        ks, kt, bs, T, n, p = 3, 3, [[CHANNEL, 16, 64], [64, 16, 64]], TIMESTEP_IN, N_NODE, 0
        A = pd.read_csv(STGCN_ADJ).values
        W = baseline_py_file.weight_matrix(A)
        L = baseline_py_file.scaled_laplacian(W)
        Lk = baseline_py_file.cheb_poly(L, ks)
        Lk = torch.Tensor(Lk.astype(np.float32)).to(device)
        adj = torch.Tensor(A.astype(np.float32)).to(device)
        adj[adj > 0] = 1
        # TODO: 半径选取多少？
        model = baseline_py_file.stgcn17(ks, kt, bs, T, n, Lk, p, adj, int(args.rho*N_NODE), quantiles, TIMESTEP_OUT)          
    if args.model == 'gwn1':
        adj_mx = baseline_py_file.load_adj(DCRNN_ADJ, ADJTYPE)
        supports = [torch.tensor(i).to(device) for i in adj_mx] if args.adj else None
        model = baseline_py_file.gwn1(device, quantiles=quantiles, num_nodes=N_NODE, in_dim=CHANNEL, supports=None,
                                      layers=int(np.log2(args.instep)) + 1).to(device)
    if args.model == 'gwn2':
        adj_mx = baseline_py_file.load_adj(DCRNN_ADJ, ADJTYPE)
        supports = [torch.tensor(i).to(device) for i in adj_mx] if args.adj else None
        model = baseline_py_file.gwn2(device, quantiles=quantiles, num_nodes=N_NODE, in_dim=CHANNEL, supports=None,
                                      layers=int(np.log2(args.instep)) + 1).to(device)
    if args.model == 'gwn3':
        adj_mx = baseline_py_file.load_adj(DCRNN_ADJ, ADJTYPE)
        supports = [torch.tensor(i).to(device) for i in adj_mx] if args.adj else None
        model = baseline_py_file.gwn3(device, quantiles=quantiles, num_nodes=N_NODE, in_dim=CHANNEL, supports=None,
                                      layers=2).to(device)
    if args.model == 'gwn4':
        adj_mx = baseline_py_file.load_adj(DCRNN_ADJ, ADJTYPE)
        supports = [torch.tensor(i).to(device) for i in adj_mx] if args.adj else None
        model = baseline_py_file.gwn4(device, quantiles=quantiles, num_nodes=N_NODE, in_dim=CHANNEL, supports=None,
                                      layers=2).to(device)
    if args.model == 'dcrnn1':
        adj_mx = baseline_py_file.load_adj(DCRNN_ADJ, ADJTYPE)
        model = baseline_py_file.dcrnn1(device, num_nodes=N_NODE, input_dim=CHANNEL, out_horizon=TIMESTEP_OUT, P=adj_mx,
                                        quantiles=quantiles).to(device)
    if args.model == 'dcrnn2':
        adj_mx = baseline_py_file.load_adj(DCRNN_ADJ, ADJTYPE)
        model = baseline_py_file.dcrnn2(quantiles, device, num_nodes=N_NODE, input_dim=CHANNEL,
                                        out_horizon=TIMESTEP_OUT, P=adj_mx).to(device)
    if args.model == 'dcrnn3':
        adj_mx = baseline_py_file.load_adj(DCRNN_ADJ, ADJTYPE)
        model = baseline_py_file.dcrnn3(quantiles, device, num_nodes=N_NODE, input_dim=CHANNEL,
                                        out_horizon=TIMESTEP_OUT, P=adj_mx).to(device)

    # 原版STGCN
    if args.model == 'stgcn3':
        ks, kt, bs, T, n, p = 3, 3, [[CHANNEL, 16, 64], [64, 16, 64]], TIMESTEP_IN, N_NODE, 0
        A = pd.read_csv(STGCN_ADJ).values
        W = baseline_py_file.weight_matrix(A)
        L = baseline_py_file.scaled_laplacian(W)
        Lk = baseline_py_file.cheb_poly(L, ks)
        Lk = torch.Tensor(Lk.astype(np.float32)).to(device)
        model = baseline_py_file.stgcn3(ks, kt, bs, T, n, Lk, p, quantiles, TIMESTEP_OUT).to(device)
        ###############################################
    ### initial the model parameters ###
    #     for p in model.parameters():
    #         if p.dim() > 1:
    #             nn.init.xavier_uniform_(p)
    #         else:
    #             nn.init.uniform_(p)
    if args.model == 'staeformer':
        model = baseline_py_file.staeformer(num_nodes=N_NODE,
        in_steps=12,
        out_steps=12,
        steps_per_day=288,
        input_dim=3,
        output_dim=3,
        input_embedding_dim=24,
        tod_embedding_dim=24,
        dow_embedding_dim=24,
        spatial_embedding_dim=0,
        adaptive_embedding_dim=80,
        feed_forward_dim=256,
        num_heads=4,
        num_layers=3,
        dropout=0.1,
        use_mixed_proj=True).to(device)
    if args.model == 'stnorm':
        model = baseline_py_file.stnorm(device, quantiles= quantiles, num_nodes= N_NODE, tnorm_bool=True, snorm_bool=True, in_dim=1,out_dim=TIMESTEP_OUT, channels=32, kernel_size=2, blocks=1, layers=4).to(device)
#####   
    if args.model == 'scinet':
        model = baseline_py_file.scinet(output_len=TIMESTEP_OUT, input_len=TIMESTEP_IN, input_dim = N_NODE, hid_size = 0.0625, num_stacks = 1,
                num_levels = 2, num_decoder_layer = 1, concat_len = 0, groups = 1, kernel = 5, dropout = 0, single_step_output_One = 0, input_len_seg = 0, positionalE = True, modified = True, RIN=False).to(device)
#         model = baseline_py_file.scinet(device,output_len=TIMESTEP_OUT, input_len=TIMESTEP_IN, input_dim = N_NODE, hid_size = 0.0625, num_stacks = 2,
#                 num_levels = 3, num_decoder_layer = 2, concat_len = 0, groups = N_NODE, kernel = 3, dropout = 0,
#                  single_step_output_One = 0, input_len_seg = 0, positionalE = False, modified = True, RIN=True).to(device)
    return model
def predictModel_stack(name, s_model, q_model,t_model, stack_model, data_iter):
    YS_truth = []
    YS_pred = []
    s_model.eval()
    q_model.eval()
    t_model.eval()
    stack_model.eval()    
    with torch.no_grad():
        for x, y in data_iter:
            
            x = x.to(device)
            y = y.to(device)
            s_ypred = s_model(x)    # [b,t,n,q]
            q_ypred = q_model(x)    # [b,t,n,q]
            t_ypred = t_model(x)    # [b,t,n,q]
            pred = torch.cat((s_ypred,q_ypred,t_ypred),dim=-1)
            ypred = stack_model(pred)
            YS_pred_batch = ypred.cpu().numpy()
            YS_truth_batch = y.cpu().numpy()
            YS_pred.append(YS_pred_batch)
            YS_truth.append(YS_truth_batch)
        YS_pred = np.vstack(YS_pred)
        YS_truth = np.vstack(YS_truth)
    # print('YS_pred shape is : ',YS_pred.shape) # (128, 1, 307, 3)
    return YS_truth, YS_pred  # [B,T,N,C]

def predictModel(name, model, data_iter):
    YS_truth = []
    YS_pred = []
    model.eval()
    with torch.no_grad():
        for x, y in data_iter:
            x = x.to(device)
            y = y.to(device)            
            if args.model == 'stgcn1' or args.model == 'stgcn2':
                YS_truth_batch = y.cpu().numpy()
                YS_truth.append(YS_truth_batch)
                XS_pred_multi_batch, YS_pred_multi_batch = [x], []
                for i in range(TIMESTEP_OUT):
                    tmp_torch = torch.cat(XS_pred_multi_batch, dim=1)[:, i:, :, :]
                    yhat = model(tmp_torch)
                    XS_pred_multi_batch.append(yhat[:, :, :, 1:2])
                    YS_pred_multi_batch.append(yhat)
                YS_pred_multi_batch = torch.cat(YS_pred_multi_batch, dim=1).cpu().numpy()
                # print('YS_pred_multi_batch shape is : ',YS_pred_multi_batch.shape) # (128, 1, 307, 3)
                YS_pred.append(YS_pred_multi_batch)
            else:
                ypred = model(x)    # [b,t,n,q]
                YS_pred_batch = ypred.cpu().numpy()
                YS_truth_batch = y.cpu().numpy()
                YS_pred.append(YS_pred_batch)
                YS_truth.append(YS_truth_batch)
        YS_pred = np.vstack(YS_pred)
        YS_truth = np.vstack(YS_truth)
    # print('YS_pred shape is : ',YS_pred.shape) # (128, 1, 307, 3)
    return YS_truth, YS_pred  # [B,T,N,C]

def model_inference(name, model, val_iter, test_iter, x_stats, save=True, with_point=False, point_prediction=False):
    val_y_truth, val_y_pred = predictModel(name, model, val_iter)
    val_y_pred = lib.Utils.z_inverse(val_y_pred, x_stats['mean'], x_stats['std'])
    # val_y_truth = lib.Utils.z_inverse(val_y_truth, x_stats['mean'], x_stats['std'])
    val_rmse, val_mae, val_mape, val_smape, val_rse, val_quantiles = \
            lib.Metrics.evaluate(val_y_truth, val_y_pred, quantiles)  # [T]
    test_y_truth, test_y_pred = predictModel(name, model, test_iter)
    test_y_pred = lib.Utils.z_inverse(test_y_pred, x_stats['mean'], x_stats['std'])
    # test_y_truth = lib.Utils.z_inverse(test_y_truth, x_stats['mean'], x_stats['std'])
    test_rmse, test_mae, test_mape, test_smape, test_rse, test_quantiles = \
            lib.Metrics.evaluate(test_y_truth, test_y_pred, quantiles)  # [T]
    if save:
        np.save(single_version_PATH + '/' + args.data + '_' + MODELNAME + '_prediction.npy', test_y_pred)
        np.save(single_version_PATH + '/' + args.data + '_' + MODELNAME + '_groundtruth.npy', test_y_truth)
    return val_rmse, val_mae, val_mape, val_smape, val_rse, val_quantiles, test_rmse, test_mae, test_mape, test_smape, test_rse, test_quantiles

def model_inference_stack(name, s_model, q_model, t_model, stack_model, val_iter, test_iter, x_stats, save=True):
    val_y_truth, val_y_pred = predictModel_stack(name, s_model, q_model, t_model, stack_model, val_iter)
    val_y_pred = lib.Utils.z_inverse(val_y_pred, x_stats['mean'], x_stats['std'])
    val_rmse, val_mae, val_mape, val_smape, val_rse, val_quantiles = \
            lib.Metrics.evaluate(val_y_truth, val_y_pred, quantiles)  # [T]
    test_y_truth, test_y_pred = predictModel_stack(name, s_model, q_model, t_model, stack_model, test_iter)
    test_y_pred = lib.Utils.z_inverse(test_y_pred, x_stats['mean'], x_stats['std'])
    # test_y_truth = lib.Utils.z_inverse(test_y_truth, x_stats['mean'], x_stats['std'])
    test_rmse, test_mae, test_mape, test_smape, test_rse, test_quantiles = \
            lib.Metrics.evaluate(test_y_truth, test_y_pred, quantiles)  # [T]    
    if save:
        np.save(single_version_PATH + '/' + args.data + '_' + MODELNAME + '_prediction.npy', test_y_pred)
        np.save(single_version_PATH + '/' + args.data + '_' + MODELNAME + '_groundtruth.npy', test_y_truth)
    return val_rmse, val_mae, val_mape, val_smape, val_rse, val_quantiles, test_rmse, test_mae, test_mape, test_smape, test_rse, test_quantiles

def curriculum_p(p,retain=0.3, T=100, num_bz=100):
    gamma = 1000/(T*N_NODE*num_bz)
    return 1-(1-retain)*np.exp(-gamma*p)

def weight_reset(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        m.reset_parameters()
        
class StackingModel(nn.Module):
    def __init__(self, input_dim):
        super(StackingModel, self).__init__()
        self.fc = nn.Linear(input_dim, 3)  # 你可以调整输出的维度
    
    def forward(self, x):
        return self.fc(x)        
        
def trainModel(name, device, trainset_loader,valset_loader,testset_loader, x_stats):
    mode = 'Train'
    print('Model Training Started ...', time.ctime())
    print('TIMESTEP_IN, TIMESTEP_OUT', TIMESTEP_IN, TIMESTEP_OUT)
    s_model = getModel(name, device).to(device)
    t_model = getModel(name, device).to(device)
    q_model = getModel(name, device).to(device)
    
    #     summary(model, (TIMESTEP_IN, N_NODE, CHANNEL), device=device)
    train_iter = trainset_loader
    val_iter = valset_loader
    test_iter = testset_loader
    torch_mean = torch.Tensor(x_stats['mean'].reshape((1, 1, -1, 1))).to(device)
    torch_std = torch.Tensor(x_stats['std'].reshape((1, 1, -1, 1))).to(device)
    p = 0
    print('LOSS is :', LOSS)
    with_point = False
    if LOSS == "masked_mae":
        criterion = lib.Utils.masked_mae
    if LOSS == "masked_mse":
        criterion = lib.Utils.masked_mse
    if LOSS == 'mse':
        criterion = nn.MSELoss()
    if LOSS == 'mae':
        criterion = nn.L1Loss()
    if LOSS == 'quantile':
        init_criterion = lib.Utils.QuantileLoss(quantiles)
        s_criterion = lib.Utils.QuantileLoss(quantiles)
        t_criterion = lib.Utils.QuantileLoss(quantiles)
    if 'combined' in LOSS:
        criterion = lib.Utils.QuantileLoss(quantiles)
        # criterion = lib.Utils.masked_mse
        with_point = True
        model = Model(model, quantiles).to(device)
    # if LOSS == 'mixed':
    #     criterion = lib.Utils.MixedLoss(ALPHA, quantiles)
    if OPTIMIZER == 'RMSprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=LEARNING_RATE)
    elif OPTIMIZER == 'Adam':
        s_optimizer = torch.optim.Adam(s_model.parameters(), weight_decay=0.0001, lr=LEARNING_RATE)
        t_optimizer = torch.optim.Adam(t_model.parameters(), weight_decay=0.0001, lr=LEARNING_RATE)
        q_optimizer = torch.optim.Adam(q_model.parameters(), weight_decay=0.0001, lr=LEARNING_RATE)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, gamma=0.1, milestones=[30],verbose=False)
    s_scheduler = torch.optim.lr_scheduler.StepLR(s_optimizer, step_size=5, gamma=0.7)
    t_scheduler = torch.optim.lr_scheduler.StepLR(t_optimizer, step_size=5, gamma=0.7)
    q_scheduler = torch.optim.lr_scheduler.StepLR(q_optimizer, step_size=5, gamma=0.7)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7
    qiter_count, titer_count, siter_count, qcl_length, tcl_length, scl_length = 1, 1, 1, 1, 1, 1
    if args.qcl:
        quantiles_init = [0.1+args.qcl_len, 0.5, 0.9-args.qcl_len] if args.qcldes else [0.1-(args.qcl_times-1)*args.qcl_len/args.qcl_times, 0.5, 1-args.qcl_len/args.qcl_times] # [0.01,0.5,0.99]
        quantiles2 = quantiles_init
        
        q_criterion = lib.Utils.QuantileLoss(quantiles2)
    iter_quantile_metric = []
    mask_rank = None
    pred_mask = -1
    y_mask = -1
    min_val_loss = {'spatial CL model':np.inf,'quantile CL model':np.inf,'temporal CL model':np.inf}
    wait =  {'spatial CL model':0,'quantile CL model':0,'temporal CL model':0}
    stop =  {'spatial CL model':None,'quantile CL model':None,'temporal CL model':None}    
    best_epoch = {'spatial CL model':None,'quantile CL model':None,'temporal CL model':None} 
    if bool(args.tcl):
        print(f"TCL target length = %d" % tcl_length)
    if bool(args.qcl):    
        print("qcl init quantile loss is",  quantiles_init)        
    for epoch in range(EPOCH):  # EPOCH
        if stop['spatial CL model']==True and stop['quantile CL model']==True and stop['temporal CL model']==True:
            break
        starttime = datetime.now()
        loss_sum, n = 0.0, 0
        s_model.train()
        q_model.train()
        t_model.train()       
        for i, (x, y) in enumerate(train_iter):
            
            x = x.to(device)
            y = y.to(device)
            if stop['spatial CL model']==None:
                s_optimizer.zero_grad()
                if siter_count < args.wait_iter: # args.wait_scl default is 1   
                    ypred = s_model(x)
                    ypred = ypred * torch_std + torch_mean
                    temp_iter_quantile_metric = lib.Utils.get_quantile_score2(ypred, y, qs=[0.1,0.5,0.9])
                    every_iter_quantile_metric = np.mean(temp_iter_quantile_metric.cpu().numpy(),axis=(0,1,3))  # torch tensor [n]
                    del temp_iter_quantile_metric
                    iter_quantile_metric.append(every_iter_quantile_metric)    
                if siter_count == args.wait_iter:
                    spatial_quantile_rank = np.mean(np.array(iter_quantile_metric),axis=0)  
                    s_model.apply(weight_reset)                  
                if siter_count >= args.wait_iter and scl_length <= args.scl_length:
                    if (siter_count-args.wait_iter) % args.scl_size == 0 and scl_length <= args.scl_length:
                        # sstop = True if scl_length==args.scl_length and (siter_count-args.wait_iter) % args.scl_size == 0 == 0 else None
                        print(f"Spatial CL target length = %d" % scl_length)   
                        mask_rank = lib.Utils.get_quantile_score3(args.descending, N_NODE, scl_length, args.scl_length, spatial_quantile_rank)
                        mask_rank = mask_rank if type(mask_rank)==int else  torch.Tensor(mask_rank).to(x.device).repeat(x.shape[0], 1, 1, 1).view(x.shape[0],1,1,x.shape[2]) 
                        # pred_mask = -1 if type(mask_rank)==int else  mask_rank.permute(0,1,3,2).repeat(1,12,1,3)
                        y_mask = -1 if type(mask_rank)==int else  mask_rank.permute(0,1,3,2).repeat(1,12,1,1)                    
                        scl_length+=1
                    ypred = s_model(x)
                    ypred = ypred * torch_std + torch_mean
                if siter_count >= args.wait_iter and scl_length > args.scl_length:
                    ypred = s_model(x)
                    ypred = ypred * torch_std + torch_mean
                if siter_count >= args.wait_iter:
#                     loss = criterion(ypred, y)
                    loss = s_criterion(ypred *  pred_mask*-1, y*  y_mask*-1) 
                else:
                    loss = s_criterion(ypred, y)    
                loss.backward()
                loss_sum += loss.item() * y.shape[0]
                n += y.shape[0]                
                s_optimizer.step()          
                siter_count += 1
            if stop['quantile CL model']==None:
                q_optimizer.zero_grad()
                ypred = q_model(x)
                ypred = ypred * torch_std + torch_mean
                if  qiter_count % args.qcl_size == 0 and qcl_length <= args.qcl_times:
                    # qstop = True if qcl_length==args.qcl_times and qiter_count % args.qcl_size == 0 else None
                    quantiles2 = [0.1+args.qcl_len-qcl_length*args.qcl_len/args.qcl_times, 0.5, 0.9-args.qcl_len+qcl_length*args.qcl_len/args.qcl_times] if args.qcldes else [0.1-(args.qcl_len-qcl_length*args.qcl_len/args.qcl_times), 0.5, 0.9+args.qcl_len-qcl_length*args.qcl_len/args.qcl_times] #  [0.01,0.5,0.99]  ---> [0.1,0.5,0.9]   or [0.01,0.5,0.99]  ---> [0.1,0.5,0.9]
                    qcl_length += 1
                    q_criterion = lib.Utils.QuantileLoss(quantiles2)
                    print("Quantile cl loss is", quantiles2)    
                loss = q_criterion(ypred, y)  
                loss.backward()
                loss_sum += loss.item() * y.shape[0]
                n += y.shape[0]                               
                q_optimizer.step()
                qiter_count += 1                    
            if stop['temporal CL model']==None:
                t_optimizer.zero_grad()
                ypred = t_model(x)      
                ypred = ypred * torch_std + torch_mean
                if titer_count % args.tcl_size == 0 and tcl_length <= TIMESTEP_OUT:
                    tcl_length += 1
                    print(f"Temporal CL length = %d" % tcl_length)
                loss = t_criterion(ypred[:, :tcl_length, :, :], y[:, :tcl_length, :, :])
                loss.backward()
                loss_sum += loss.item() * y.shape[0]
                n += y.shape[0]                
                t_optimizer.step()                
                titer_count += 1 
        s_scheduler.step() if stop['spatial CL model']==None else None
        q_scheduler.step() if stop['quantile CL model']==None else None
        t_scheduler.step() if stop['temporal CL model']==None else None
        if epoch % PRINT_EPOCH == 0:
            train_loss = loss_sum / n
            # 原版STGCN的点预测
            for model_type, model in zip(['spatial CL model','quantile CL model','temporal CL model'],[s_model,q_model,t_model]):
                val_rmse, val_mae, val_mape, val_smape, val_rse, val_quantiles, \
                test_rmse, test_mae, test_mape, test_smape, test_rse, test_quantiles = model_inference(name, model,
                                                                                                       val_iter,
                                                                                                       test_iter,
                                                                                                       x_stats,
                                                                                                       save=False,
                                                                                                       with_point=False)
                print(f'In {model_type}  Epoch {epoch}: ')
                #             print("| 3  Horizon | RMSE: %.3f, %.3f; MAE: %.3f, %.3f; MAPE: %.3f, %.3f; SMAPE: %.3f, %.3f; RSE: %.3f, %.3f; Q10: %.3f, %.3f; Q90: %.3f, %.3f;" % (
                #             val_rmse[2], test_rmse[2], val_mae[2], test_mae[2], val_mape[2], test_mape[2], val_smape[2], test_smape[2], val_rse[2], test_rse[2], val_quantiles[2,0], test_quantiles[2,0], val_quantiles[2,2], test_quantiles[2,2]))
                print(
                    "| 3  Horizon | RMSE: %.3f, %.3f; MAE: %.3f, %.3f; MAPE: %.3f, %.3f; Q10: %.3f, %.3f; Q50: %.3f, %.3f; Q90: %.3f, %.3f;" % (
                        val_rmse[2], test_rmse[2], val_mae[2], test_mae[2], val_mape[2], test_mape[2], val_quantiles[2, 0],
                        test_quantiles[2, 0], val_quantiles[2, 1], test_quantiles[2, 1], val_quantiles[2, 2],
                        test_quantiles[2, 2]))
                print(
                    "| 6  Horizon | RMSE: %.3f, %.3f; MAE: %.3f, %.3f; MAPE: %.3f, %.3f; Q10: %.3f, %.3f; Q50: %.3f, %.3f; Q90: %.3f, %.3f;" % (
                        val_rmse[5], test_rmse[5], val_mae[5], test_mae[5], val_mape[5], test_mape[5], val_quantiles[5, 0],
                        test_quantiles[5, 0], val_quantiles[5, 1], test_quantiles[5, 1], val_quantiles[5, 2],
                        test_quantiles[5, 2]))
                print(
                    "| 12 Horizon | RMSE: %.3f, %.3f; MAE: %.3f, %.3f; MAPE: %.3f, %.3f; Q10: %.3f, %.3f; Q50: %.3f, %.3f; Q90: %.3f, %.3f;" % (
                        val_rmse[11], test_rmse[11], val_mae[11], test_mae[11], val_mape[11], test_mape[11],
                        val_quantiles[11, 0], test_quantiles[11, 0], val_quantiles[11, 1], test_quantiles[11, 1],
                        val_quantiles[11, 2], test_quantiles[11, 2]))

                with open(single_version_PATH + '/' + name + '_' + model_type + '_log.txt', 'a') as f:
                    f.write(f'Epoch {epoch}: \n')
                    f.write(
                        "| 3  Horizon | RMSE: %.3f, %.3f; MAE: %.3f, %.3f; MAPE: %.3f, %.3f; Q10: %.3f, %.3f; Q50: %.3f, %.3f; Q90: %.3f, %.3f;\n" % (
                            val_rmse[2], test_rmse[2], val_mae[2], test_mae[2], val_mape[2], test_mape[2],
                            val_quantiles[2, 0], test_quantiles[2, 0], val_quantiles[2, 1], test_quantiles[2, 1],
                            val_quantiles[2, 2], test_quantiles[2, 2]))
                    f.write(
                        "| 6  Horizon | RMSE: %.3f, %.3f; MAE: %.3f, %.3f; MAPE: %.3f, %.3f; Q10: %.3f, %.3f; Q50: %.3f, %.3f; Q90: %.3f, %.3f;\n" % (
                            val_rmse[5], test_rmse[5], val_mae[5], test_mae[5], val_mape[5], test_mape[5],
                            val_quantiles[5, 0], test_quantiles[5, 0], val_quantiles[5, 1], test_quantiles[5, 1],
                            val_quantiles[5, 2], test_quantiles[5, 2]))
                    f.write(
                        "| 12 Horizon | RMSE: %.3f, %.3f; MAE: %.3f, %.3f; MAPE: %.3f, %.3f; Q10: %.3f, %.3f; Q50: %.3f, %.3f; Q90: %.3f, %.3f;\n" % (
                            val_rmse[11], test_rmse[11], val_mae[11], test_mae[11], val_mape[11], test_mape[11],
                            val_quantiles[11, 0], test_quantiles[11, 0], val_quantiles[11, 1], test_quantiles[11, 1],
                            val_quantiles[11, 2], test_quantiles[11, 2]))
                #             val_loss, test_loss = val_mae, test_mae
                val_loss, test_loss = val_quantiles, test_quantiles                
                        
#     min_val_loss = {'spatial CL model':np.inf,'quantile CL model':np.inf,'temporal CL model':np.inf}
#     wait =  {'spatial CL model':0,'quantile CL model':0,'temporal CL model':0}
#     stop =  {'spatial CL model':None,'quantile CL model':None,'temporal CL model':None}  
#     best_epoch = {'spatial CL model':None,'quantile CL model':None,'temporal CL model':None} 
    
                if scl_length >= args.scl_length and tcl_length > TIMESTEP_OUT and qcl_length >= args.qcl_times:
                    if stop[model_type] == None:
                        if np.mean(val_loss) < min_val_loss[model_type]:
                            wait[model_type] = 0
                            min_val_loss[model_type] = np.mean(val_loss)
                            best_epoch[model_type] = epoch
                            torch.save(model.state_dict(), single_version_PATH + '/' + name + '_' + model_type + '.pt')
                        else:
                            wait[model_type] += 1
                        if wait[model_type] == PATIENCE:
                            print('*'*27)
                            print('*'*27)
                            print(model_type, ' Early stopping at epoch:', epoch)
                            print('*'*27)
                            print('*'*27)
                            stop[model_type] = True
                    
        endtime = datetime.now()
        epoch_time = (endtime - starttime).seconds
        print("epoch", epoch, "time used:", epoch_time, " seconds ", "train loss:", np.around(train_loss, 3),
              "val loss:", np.around(np.mean(val_loss), 6), "test loss:", np.around(np.mean(test_loss), 6))
        with open(single_version_PATH + '/' + name + '_log.txt', 'a') as f:
            f.write("%s, %d, %s, %d, %s, %s, %.3f, %s, %.3f, %s, %.3f\n" % (
                "epoch", epoch, "time used", epoch_time, "seconds", "train loss", train_loss, "validation loss:",
                np.mean(val_loss), "test loss:", np.mean(test_loss)))
    s_model.load_state_dict(torch.load(single_version_PATH + '/' + name + '_spatial CL model.pt'))
    q_model.load_state_dict(torch.load(single_version_PATH + '/' + name + '_quantile CL model.pt'))
    t_model.load_state_dict(torch.load(single_version_PATH + '/' + name + '_temporal CL model.pt'))
    print('*'*27)
    print('*'*27)    
    print('stacking start: ')
    print('*'*27)
    print('*'*27)       
    stack_model = StackingModel(3*3).to(device)
    stack_optimizer = torch.optim.Adam(stack_model.parameters(), weight_decay=0.0001, lr=0.005)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, gamma=0.1, milestones=[30],verbose=False)
    stack_scheduler = torch.optim.lr_scheduler.StepLR(stack_optimizer, step_size=10, gamma=0.7)
    stack_min_val_loss = np.inf
    stack_wait = 0
    stack_criterion = lib.Utils.QuantileLoss(quantiles)
    for epoch in range(EPOCH):
        stack_model.train()
        starttime = datetime.now()
        loss_sum, n = 0.0, 0
        s_model.eval()
        q_model.eval()
        t_model.eval()
        p += 1
        # TODO: retain 和 T 分别设置多少
        pp = curriculum_p(p, retain=args.retain, T=args.clweithT, num_bz=(len(train_iter) + len(val_iter)) / BATCHSIZE)           
        for i, (x, y) in enumerate(train_iter): 
            stack_optimizer.zero_grad()
            
            x = x.to(device)
            y = y.to(device)
            with torch.no_grad():
                s_pred = s_model(x)
                q_pred = q_model(x)
                t_pred = t_model(x)
            pred = torch.cat((s_pred,q_pred,t_pred),dim=-1)
            y_pred = stack_model(pred)
            y_pred = y_pred * torch_std + torch_mean
            loss = stack_criterion(y_pred, y) 
            loss.backward()
            stack_optimizer.step()
        stack_scheduler.step()
        if epoch % PRINT_EPOCH == 0:
            val_rmse, val_mae, val_mape, val_smape, val_rse, val_quantiles, \
                    test_rmse, test_mae, test_mape, test_smape, test_rse, test_quantiles = model_inference_stack(name, s_model, q_model,t_model, stack_model,
                                                                                                           val_iter,
                                                                                                           test_iter,
                                                                                                           x_stats,
                                                                                                           save=False)
            print(f'In Stacking model  Epoch {epoch}: ')
            #             print("| 3  Horizon | RMSE: %.3f, %.3f; MAE: %.3f, %.3f; MAPE: %.3f, %.3f; SMAPE: %.3f, %.3f; RSE: %.3f, %.3f; Q10: %.3f, %.3f; Q90: %.3f, %.3f;" % (
            #             val_rmse[2], test_rmse[2], val_mae[2], test_mae[2], val_mape[2], test_mape[2], val_smape[2], test_smape[2], val_rse[2], test_rse[2], val_quantiles[2,0], test_quantiles[2,0], val_quantiles[2,2], test_quantiles[2,2]))
            print(
                "| 3  Horizon | RMSE: %.3f, %.3f; MAE: %.3f, %.3f; MAPE: %.3f, %.3f; Q10: %.3f, %.3f; Q50: %.3f, %.3f; Q90: %.3f, %.3f;" % (
                    val_rmse[2], test_rmse[2], val_mae[2], test_mae[2], val_mape[2], test_mape[2], val_quantiles[2, 0],
                    test_quantiles[2, 0], val_quantiles[2, 1], test_quantiles[2, 1], val_quantiles[2, 2],
                    test_quantiles[2, 2]))
            print(
                "| 6  Horizon | RMSE: %.3f, %.3f; MAE: %.3f, %.3f; MAPE: %.3f, %.3f; Q10: %.3f, %.3f; Q50: %.3f, %.3f; Q90: %.3f, %.3f;" % (
                    val_rmse[5], test_rmse[5], val_mae[5], test_mae[5], val_mape[5], test_mape[5], val_quantiles[5, 0],
                    test_quantiles[5, 0], val_quantiles[5, 1], test_quantiles[5, 1], val_quantiles[5, 2],
                    test_quantiles[5, 2]))
            print(
                "| 12 Horizon | RMSE: %.3f, %.3f; MAE: %.3f, %.3f; MAPE: %.3f, %.3f; Q10: %.3f, %.3f; Q50: %.3f, %.3f; Q90: %.3f, %.3f;" % (
                    val_rmse[11], test_rmse[11], val_mae[11], test_mae[11], val_mape[11], test_mape[11],
                    val_quantiles[11, 0], test_quantiles[11, 0], val_quantiles[11, 1], test_quantiles[11, 1],
                    val_quantiles[11, 2], test_quantiles[11, 2]))

            with open(single_version_PATH + '/' + name + '_' + 'stacking' + '_log.txt', 'a') as f:
                f.write(f'Epoch {epoch}: \n')
                f.write(
                    "| 3  Horizon | RMSE: %.3f, %.3f; MAE: %.3f, %.3f; MAPE: %.3f, %.3f; Q10: %.3f, %.3f; Q50: %.3f, %.3f; Q90: %.3f, %.3f;\n" % (
                        val_rmse[2], test_rmse[2], val_mae[2], test_mae[2], val_mape[2], test_mape[2],
                        val_quantiles[2, 0], test_quantiles[2, 0], val_quantiles[2, 1], test_quantiles[2, 1],
                        val_quantiles[2, 2], test_quantiles[2, 2]))
                f.write(
                    "| 6  Horizon | RMSE: %.3f, %.3f; MAE: %.3f, %.3f; MAPE: %.3f, %.3f; Q10: %.3f, %.3f; Q50: %.3f, %.3f; Q90: %.3f, %.3f;\n" % (
                        val_rmse[5], test_rmse[5], val_mae[5], test_mae[5], val_mape[5], test_mape[5],
                        val_quantiles[5, 0], test_quantiles[5, 0], val_quantiles[5, 1], test_quantiles[5, 1],
                        val_quantiles[5, 2], test_quantiles[5, 2]))
                f.write(
                    "| 12 Horizon | RMSE: %.3f, %.3f; MAE: %.3f, %.3f; MAPE: %.3f, %.3f; Q10: %.3f, %.3f; Q50: %.3f, %.3f; Q90: %.3f, %.3f;\n" % (
                        val_rmse[11], test_rmse[11], val_mae[11], test_mae[11], val_mape[11], test_mape[11],
                        val_quantiles[11, 0], test_quantiles[11, 0], val_quantiles[11, 1], test_quantiles[11, 1],
                        val_quantiles[11, 2], test_quantiles[11, 2]))
            #             val_loss, test_loss = val_mae, test_mae
            val_loss, test_loss = val_quantiles, test_quantiles                
            print("----------val loss is : ", np.around(np.mean(val_loss), 6), '----------test loss is : ',np.around(np.mean(test_loss), 6))
    #     min_val_loss = {'spatial CL model':np.inf,'quantile CL model':np.inf,'temporal CL model':np.inf}
    #     wait =  {'spatial CL model':0,'quantile CL model':0,'temporal CL model':0}
    #     stop =  {'spatial CL model':None,'quantile CL model':None,'temporal CL model':None}  
    #     best_epoch = {'spatial CL model':None,'quantile CL model':None,'temporal CL model':None} 

            if np.mean(val_loss) < stack_min_val_loss:
                stack_wait = 0
                stack_min_val_loss = np.mean(val_loss)
                stack_best_epoch = epoch
                torch.save(stack_model.state_dict(), single_version_PATH + '/' + name + '_' + 'stack_model.pt')
            else:
                stack_wait += 1
            if stack_wait == PATIENCE:
                print('*'*27)
                print('*'*27)
                print('Stack Model Early stopping at epoch:', epoch)
                print('*'*27)
                print('*'*27)
                break
    
    stack_model.load_state_dict(torch.load(single_version_PATH + '/' + name + '_' + 'stack_model.pt'))
    val_rmse, val_mae, val_mape, val_smape, val_rse, val_quantiles, \
                test_rmse, test_mae, test_mape, test_smape, test_rse, test_quantiles = model_inference_stack(name, s_model, q_model,t_model, stack_model,
                                                                                                       val_iter,
                                                                                                       test_iter,
                                                                                                       x_stats,
                                                                                                       save=False)
    print('Model ', name, ' Best Results:')
    print(f'Epoch {best_epoch}: ')
    head = "%-8s %-8s %-8s %-8s %-8s %-8s %-8s %-8s %-8s %-8s %-8s %-8s %-8s %-8s %-8s %-8s %-8s %-8s"
    row = "%-8.3f %-8.3f %-8.3f %-8.3f %-8.3f %-8.3f %-8.3f %-8.3f %-8.3f %-8.3f %-8.3f %-8.3f %-8.3f %-8.3f %-8.3f %-8.3f %-8.3f %-8.3f"
    print(head % (
        "=", "=", "Horizon", "3", "=", "=", "=", "=", "Horizon", "6", "=", "=", "=", "=", "Horizon", "12", "=", "="))
    print(head % (
        "RMSE", "MAE", "MAPE", "Q10", "Q50", "Q90", "RMSE", "MAE", "MAPE", "Q10", "Q50", "Q90", "RMSE", "MAE", "MAPE",
        "Q10", "Q50", "Q90"))
    print(row % (
        test_rmse[2], test_mae[2], test_mape[2], test_quantiles[2, 0], test_quantiles[2, 1], test_quantiles[2, 2],
        test_rmse[5], test_mae[5], test_mape[5], test_quantiles[5, 0], test_quantiles[5, 1], test_quantiles[5, 2],
        test_rmse[11], test_mae[11], test_mape[11], test_quantiles[11, 0], test_quantiles[11, 1],
        test_quantiles[11, 2]))
    print('Model Training Ended ...', time.ctime())

    head += '\n'
    row += '\n'


#     with open(single_version_PATH + '/' + name + '_log.txt', 'a') as f:
#         f.write(f'Model {name} Best Results:\n')
#         f.write(f'Epoch {best_epoch}: \n')
#         f.write(head % ("Horizon", "RMSE", "MAE", "MAPE", "Q10", "Q50", "Q90"))
#         f.write(row % (
#             3, test_rmse[2], test_mae[2], test_mape[2], test_quantiles[2, 0], test_quantiles[2, 1],
#             test_quantiles[2, 2]))
#         f.write(row % (
#             6, test_rmse[5], test_mae[5], test_mape[5], test_quantiles[5, 0], test_quantiles[5, 1],
#             test_quantiles[5, 2]))
#         f.write(row % (12, test_rmse[11], test_mae[11], test_mape[11], test_quantiles[11, 0], test_quantiles[11, 1],
#                        test_quantiles[11, 2]))


def multi_version_test(name, device, train, versions):
    mode = 'multi version test'
    print('Model Testing Started ...', time.ctime())
    print('INPUT_STEP, PRED_STEP', TIMESTEP_IN, TIMESTEP_OUT)
    s_model = getModel(name, device).to(device)
    t_model = getModel(name, device).to(device)
    q_model = getModel(name, device).to(device)
    stack_model = StackingModel(3*3).to(device)
    if LOSS == 'combined':
        criterion = lib.Utils.QuantileLoss(quantiles)
        # criterion = lib.Utils.masked_mse
        with_point = True
        model = Model(model, quantiles).to(device)
    rmse_all, mae_all, mape_all, smape_all, rse_all, quantiles_all = np.zeros(
        (len(train), len(versions), TIMESTEP_OUT)), np.zeros((len(train), len(versions), TIMESTEP_OUT)), \
                                                                     np.zeros((len(train), len(versions),
                                                                               TIMESTEP_OUT)), np.zeros(
        (len(train), len(versions), TIMESTEP_OUT)), np.zeros((len(train), len(versions), TIMESTEP_OUT)), \
                                                                     np.zeros((len(train), len(versions), TIMESTEP_OUT,
                                                                               len(quantiles)))  # [V,T]
    #     val_iter = torch_data_loader(device, data, data_type='val', shuffle=True)
    #     test_iter = torch_data_loader(device, data, data_type='test', shuffle=False)
    for train_ind, tr in enumerate(train):
        which_train = tr
        trainset_loader,val_iter,test_iter, x_stats = get_dataloaders_from_index_data(data_path,tod=True,dow=True, batch_size=BATCHSIZE)            
#         torch_mean = torch.Tensor(x_stats['mean'].reshape((1, 1, -1, 1))).to(device)
#         torch_std = torch.Tensor(x_stats['std'].reshape((1, 1, -1, 1))).to(device)           
        print('*' * 40)
        print('*' * 40)
        print('Under Train Strategy --- ', tr, ' ---:')
        for ind, v_ in enumerate(versions):
            print('--- version ', v_, ' evaluation start ---')
            multi_test_PATH = \
                "./save/{}_{}_in{}_out{}_addtime{}_adj{}_lr{}_hc{}_train{}_val{}_test{}_seed{}_loss{}_version{}_note:{}".format(
                    DATANAME, args.model, args.instep, args.outstep, args.addtime, args.adj, LEARNING_RATE, args.hc,
                    which_train,
                    VAL, TEST, SEED, LOSS, v_, args.note)
            if os.path.isfile(multi_test_PATH + '/' + name + '_stack_model.pt'):
                s_model.load_state_dict(torch.load(multi_test_PATH + '/' + name + '_spatial CL model.pt', map_location=device))
                q_model.load_state_dict(torch.load(multi_test_PATH + '/' + name + '_quantile CL model.pt', map_location=device))
                t_model.load_state_dict(torch.load(multi_test_PATH + '/' + name + '_temporal CL model.pt', map_location=device))  
                stack_model.load_state_dict(torch.load(multi_test_PATH + '/' + name + '_' + 'stack_model.pt', map_location=device))
                print("file exists: ", multi_test_PATH)
            else:
                print("file not exist", multi_test_PATH)
                break
            print('*' * 20)
            print(f'Version: {v_} Start Testing :')
            val_rmse, val_mae, val_mape, val_smape, val_rse, val_quantiles, \
            test_rmse, test_mae, test_mape, test_smape, test_rse, test_quantiles = model_inference_stack(name, s_model, q_model,t_model, stack_model,
                                                                                                   val_iter, test_iter,
                                                                                                   x_stats,
                                                                                                   save=True)  # [T]
            if len(versions) == 1:
                rmse_all[train_ind, 0], mae_all[train_ind, 0], mape_all[train_ind, 0], smape_all[train_ind, 0], rse_all[
                    train_ind, 0], quantiles_all[train_ind, 0] \
                    = test_rmse, test_mae, test_mape, test_smape, test_rse, test_quantiles
            else:
                rmse_all[train_ind, v_], mae_all[train_ind, v_], mape_all[train_ind, v_], smape_all[train_ind, v_], \
                rse_all[train_ind, v_], quantiles_all[train_ind, v_] \
                    = test_rmse, test_mae, test_mape, test_smape, test_rse, test_quantiles
            print(
                "| 3  Horizon | RMSE: %.3f, %.3f; MAE: %.3f, %.3f; MAPE: %.3f, %.3f; Q10: %.3f, %.3f; Q50: %.3f, %.3f; Q90: %.3f, %.3f;" % (
                    val_rmse[2], test_rmse[2], val_mae[2], test_mae[2], val_mape[2], test_mape[2], val_quantiles[2, 0],
                    test_quantiles[2, 0], val_quantiles[2, 1], test_quantiles[2, 1], val_quantiles[2, 2],
                    test_quantiles[2, 2]))
            print(
                "| 6  Horizon | RMSE: %.3f, %.3f; MAE: %.3f, %.3f; MAPE: %.3f, %.3f; Q10: %.3f, %.3f; Q50: %.3f, %.3f; Q90: %.3f, %.3f;" % (
                    val_rmse[5], test_rmse[5], val_mae[5], test_mae[5], val_mape[5], test_mape[5], val_quantiles[5, 0],
                    test_quantiles[5, 0], val_quantiles[5, 1], test_quantiles[5, 1], val_quantiles[5, 2],
                    test_quantiles[5, 2]))
            print(
                "| 12 Horizon | RMSE: %.3f, %.3f; MAE: %.3f, %.3f; MAPE: %.3f, %.3f; Q10: %.3f, %.3f; Q50: %.3f, %.3f; Q90: %.3f, %.3f;" % (
                    val_rmse[11], test_rmse[11], val_mae[11], test_mae[11], val_mape[11], test_mape[11],
                    val_quantiles[11, 0], test_quantiles[11, 0], val_quantiles[11, 1], test_quantiles[11, 1],
                    val_quantiles[11, 2], test_quantiles[11, 2]))
            print('--- version ', v_, ' evaluation end ---')
            print('')
    #     np.save(multi_version_PATH + '/' + MODELNAME + '_groundtruth.npy', y_truth)  # [V,samples,T,N]
    #     np.save(multi_version_PATH + '/' + MODELNAME + '_prediction.npy', y_pred)
    rmse = np.mean(rmse_all, axis=(0, 1))  # [train, V, T]  -> [T]  np.mean(mse_all, axis=(0,1))
    mae = np.mean(mae_all, axis=(0, 1))
    mape = np.mean(mape_all, axis=(0, 1))
    smape = np.mean(smape_all, axis=(0, 1))
    rse = np.mean(rse_all, axis=(0, 1))
    total_quantiles = np.mean(quantiles_all, axis=(0, 1))  # [len(quantile)]
    print('*' * 40)
    print('*' * 40)
    print('*' * 40)
    print('Results in Test Dataset in Each Horizon with All Version Average:')
    print(args.model, ' :')
    head = "%-8s %-8s %-8s %-8s %-8s %-8s %-8s %-8s %-8s %-8s %-8s %-8s %-8s %-8s %-8s %-8s %-8s %-8s"
    row = "%-8.3f %-8.3f %-8.3f %-8.3f %-8.3f %-8.3f %-8.3f %-8.3f %-8.3f %-8.3f %-8.3f %-8.3f %-8.3f %-8.3f %-8.3f %-8.3f %-8.3f %-8.3f"
    print(head % (
        "=", "=", "Horizon", "3", "=", "=", "=", "=", "Horizon", "6", "=", "=", "=", "=", "Horizon", "12", "=", "="))
    print(head % (
        "RMSE", "MAE", "MAPE", "Q10", "Q50", "Q90", "RMSE", "MAE", "MAPE", "Q10", "Q50", "Q90", "RMSE", "MAE", "MAPE",
        "Q10", "Q50", "Q90"))
    print(row % (
        rmse[2], mae[2], mape[2], total_quantiles[2, 0], total_quantiles[2, 1], total_quantiles[2, 2], rmse[5], mae[5],
        mape[5], total_quantiles[5, 0], total_quantiles[5, 1], total_quantiles[5, 2], rmse[11], mae[11], mape[11],
        total_quantiles[11, 0], total_quantiles[11, 1],
        total_quantiles[11, 2]))
    print('Model Multi Version Testing Ended ...', time.ctime())
    print("*" * 40)
    print("*" * 40)
    # print("Overleaf Format in table 2 ----  Version Average in test dataset: ")


#         print(" &   MAPE   &   MAE  &   RMSE &   MAPE   &   MAE  &   RMSE &   MAPE   &   MAE  &  RMSE  &")
#         #     GWN\_8\_1\_1 & 56.755 & 30.298 & 20.360\% & 82.327 & 40.783 & 27.326\% & 114.689 & 54.537 & 36.511\%
#         if args.data == 'exchangerate':
#             print(" & {:.6f}\% & {:.6f} & {:.6f} & {:.6f}\% & {:.6f} & {:.6f} & {:.6f}\% & {:.6f} & {:.6f} &  \\\\"
#                   .format(mape[0], mae[0], rmse[0], mape[1], mae[1], rmse[1], mape[2], mae[2], rmse[2]))
#         else:
#             print(" & {:.3f}\% & {:.3f} & {:.3f} & {:.3f}\% & {:.3f} & {:.3f} & {:.3f}\% & {:.3f} & {:.3f} &  \\\\"
#                   .format(mape[0], mae[0], rmse[0], mape[1], mae[1], rmse[1], mape[2], mae[2], rmse[2]))

def main():
    # timestamp, data = data_preprocess(args.data)
    if args.mode == 'train':  # train and test in single version
        if not os.path.exists(single_version_PATH):
            os.makedirs(single_version_PATH)
        print(single_version_PATH, 'training started', time.ctime())
        model_path = './model/' + args.model + '.py'
        shutil.copy2(model_path, single_version_PATH)
        trainset_loader,valset_loader,testset_loader,x_stats = get_dataloaders_from_index_data(data_path,tod=True,dow=True, batch_size=BATCHSIZE)    
        trainModel(MODELNAME, device, trainset_loader,valset_loader,testset_loader, x_stats)

    if args.mode == 'eval0':  # eval in sing`le version
        print('single version ', args.version, ' testing started', time.ctime())
        multi_version_test(MODELNAME, device, train=[args.train], versions=[args.version])  #
    if args.mode == 'eval':  # eval in multi version
        #         print(multi_version_PATH, 'multi version testing started', time.ctime())
        multi_version_test(MODELNAME, device, train=[args.train], versions=np.arange(0, 5))  #
    # if args.mode == 'all':
    #     if not os.path.exists(multi_version_PATH):
    #         os.makedirs(multi_version_PATH)
    #     multi_version_test(MODELNAME, device, train=[0.7], versions=np.arange(0, 5))  #


if __name__ == '__main__':
    main()
