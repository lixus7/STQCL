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
parser.add_argument('-version', type=str, default='test', help='train version')
parser.add_argument('-note', type=str, default='', help='additional information')
parser.add_argument('-instep', type=int, default=12, help='input step')
parser.add_argument('-outstep', type=int, default=12, help='predict step')
parser.add_argument('-hc', type=int, default=32, help='hidden channel')
parser.add_argument('-batch', type=int, default=8, help='batch size')
parser.add_argument('-epoch', type=int, default=500, help='training epochs')
parser.add_argument('-gs', type=int, default=1, help='mean,std')
parser.add_argument('-addtime', default=False, action="store_true", help='Add timestamp')
parser.add_argument('-adj', default=False, action="store_true", help='Add adj')
parser.add_argument('-mode', type=str, default='train', help='train or eval0')
parser.add_argument("-debug", "-de", default=False, action="store_true")
parser.add_argument('-data', type=str, default='pems04', help='pems03,04,07,08')
parser.add_argument('-train', type=float, default=0.6, help='train data: 0.8,0.7,0.6,0.5')
parser.add_argument('-scaler', type=str, default='zscore', help='data scaler process type, zscore or minmax')
parser.add_argument('-cuda', type=int, default=6, help='cuda device number')
parser.add_argument('-loss', type=str, default='quantile', help='loss function, combine1, combine2, combine3')
parser.add_argument('-seed', default=None, help='torch & numpy seed')
parser.add_argument('-sche', default=False, action="store_true", help='add schedular')
parser.add_argument('-cl_size', type=int, default=2500, help='cl_size')
parser.add_argument('-warm_epoch', type=int, default=5, help='cl_size')
parser.add_argument('-cl', default=False, action="store_true", help='if add cl')
parser.add_argument('-cl2', default=False, action="store_true", help='if add cl in pointforecasting stage')
args = parser.parse_args()  # python
# args = parser.parse_args(args=[])    # jupyter notebook
device = torch.device("cuda:{}".format(args.cuda)) if torch.cuda.is_available() else torch.device("cpu")
################# Global Parameters setting ####################### 
IFGS = True if args.gs == 1 else False
DATANAME = args.data
quantiles = [0.1, 0.5, 0.9]
MODELNAME = args.model
BATCHSIZE = args.batch
EPOCH = args.epoch
if args.debug:
    EPOCH = 20
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
PATIENCE = 10
if args.model=='scinet':
    PATIENCE = 30
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

def data_preprocess(whichdata):
    if whichdata == 'pems04':
        data = np.load('./data/PEMS04/PEMS04.npz')['data'][:, :, 0]  # [samples,nodes]
        start = '2018-01-01 00:00:00'
        end = '2018-2-28 23:59:00'
        freq = '300s'
        pdates = pd.date_range(start=start, end=end, freq=freq)
        print('pdate shape is: ', pdates.shape)
        time = pd.DataFrame(pdates, columns=['date'])
    # if whichdata=='metrla':
    # data = pd.read_csv('')
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
    total_num = int(data.shape[0] * (train + VAL + TEST))
    trainval_data, test_data = [], []  # TV : Train and Val
    if if_stats == True:
        if IFGS:
            train_mean, train_std = np.mean(data[:trainval_num, :, 0], axis=0), np.std(data[:trainval_num, :, 0],
                                                                                       axis=0)
        else:
            train_mean, train_std = np.mean(data[:trainval_num, :, 0], axis=(0, 1)), np.std(data[:trainval_num, :, 0],
                                                                                            axis=(0, 1))
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
    if ifaddtime == True:
        timestamp = (timestamp - timestamp.astype("datetime64[D]")) / np.timedelta64(1, "D")
        sca_seq_time, _ = seq(timestamp, train, if_stats=False)  # [samples,N] -> {train,val,test} [B,T,N,C]

    for key in seq_data.keys():
        #         print(key ,seq_data[key][:, 0:TIMESTEP_IN, :, 0:1].shape)
        seq_data[key][:, 0:TIMESTEP_IN, :, 0:1] = lib.Utils.z_score(seq_data[key][:, 0:TIMESTEP_IN, :, 0:1],
                                                                    x_stats['mean'], x_stats['std'])
        if ifaddtime == True:
            seq_data[key] = np.concatenate((seq_data[key], sca_seq_time[key]), axis=-1)  # [B,T,N,C] -> [B,T,N,C+1]
        if args.debug:
            seq_data[key] = seq_data[key][:128]
    return seq_data, x_stats


def torch_data_loader(device, data, data_type, shuffle=True):
    x = torch.Tensor(data[data_type][:, 0:TIMESTEP_IN, :, :]).to(device)  # [B,T=TIMESTEP_IN,N,C]
    y = torch.Tensor(data[data_type][:, TIMESTEP_IN:TIMESTEP_IN + TIMESTEP_OUT, :, :]).to(
        device)  # [B,T=TIMESTEP_OUT,N,C]
    data = torch.utils.data.TensorDataset(x, y)
    data_iter = torch.utils.data.DataLoader(data, BATCHSIZE, shuffle=shuffle)
    return data_iter


def getModel(name, device):
    ### load different baseline model.py  ###
    model_path = './model/' + args.model + '.py'  # AGCRN.py 的路径
    loader = importlib.machinery.SourceFileLoader('baseline_py_file', model_path)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    baseline_py_file = importlib.util.module_from_spec(spec)
    loader.exec_module(baseline_py_file)
    ########## select the baseline model ##########
    ADJTYPE = 'doubletransition'
    if args.model == 'stgcn1':
        ks, kt, bs, T, n, p = 3, 3, [[CHANNEL, 16, 64], [64, 16, 64]], TIMESTEP_IN, N_NODE, 0
        A = pd.read_csv(STGCN_ADJ).values
        W = baseline_py_file.weight_matrix(A)
        L = baseline_py_file.scaled_laplacian(W)
        Lk = baseline_py_file.cheb_poly(L, ks)
        Lk = torch.Tensor(Lk.astype(np.float32)).to(device)
        model = baseline_py_file.stgcn1(ks, kt, bs, T, n, Lk, p, quantiles).to(device)
#     if args.model == 'stgcn2':
#         ks, kt, bs, T, n, p = 3, 3, [[CHANNEL, 16, 64], [64, 16, 64]], TIMESTEP_IN, N_NODE, 0
#         A = pd.read_csv(STGCN_ADJ).values
#         W = baseline_py_file.weight_matrix(A)
#         L = baseline_py_file.scaled_laplacian(W)
#         Lk = baseline_py_file.cheb_poly(L, ks)
#         Lk = torch.Tensor(Lk.astype(np.float32)).to(device)
#         model = baseline_py_file.stgcn2(ks, kt, bs, T, n, Lk, p, quantiles, TIMESTEP_OUT).to(device)
    if args.model == 'stgcn3':
        ks, kt, bs, T, n, p = 3, 3, [[CHANNEL, 16, 64], [64, 16, 64]], TIMESTEP_IN, N_NODE, 0
        A = pd.read_csv(STGCN_ADJ).values
        W = baseline_py_file.weight_matrix(A)
        L = baseline_py_file.scaled_laplacian(W)
        Lk = baseline_py_file.cheb_poly(L, ks)
        Lk = torch.Tensor(Lk.astype(np.float32)).to(device)
        model = baseline_py_file.stgcn3(ks, kt, bs, T, n, Lk, p, quantiles, TIMESTEP_OUT).to(device)
    if args.model == 'stgcn4':
        ks, kt, bs, T, n, p = 3, 3, [[CHANNEL, 16, 64], [64, 16, 64]], TIMESTEP_IN, N_NODE, 0
        A = pd.read_csv(STGCN_ADJ).values
        W = baseline_py_file.weight_matrix(A)
        L = baseline_py_file.scaled_laplacian(W)
        Lk = baseline_py_file.cheb_poly(L, ks)
        Lk = torch.Tensor(Lk.astype(np.float32)).to(device)
        model = baseline_py_file.stgcn4(ks, kt, bs, T, n, Lk, p, quantiles).to(device)
#     if args.model == 'gwn1':
#         adj_mx = baseline_py_file.load_adj(DCRNN_ADJ, ADJTYPE)
#         supports = [torch.tensor(i).to(device) for i in adj_mx] if args.adj else None
#         model = baseline_py_file.gwn1(device, quantiles=quantiles, num_nodes=N_NODE, in_dim=CHANNEL, supports=None,
#                                       layers=int(np.log2(args.instep)) + 1).to(device)
    if args.model == 'gwn2':
        adj_mx = baseline_py_file.load_adj(DCRNN_ADJ, ADJTYPE)
        supports = [torch.tensor(i).to(device) for i in adj_mx] if args.adj else None
        model = baseline_py_file.gwn2(device, quantiles=quantiles, num_nodes=N_NODE, in_dim=CHANNEL, supports=None,
                                      layers=int(np.log2(args.instep)) + 1).to(device)
#     if args.model == 'gwn3':
#         adj_mx = baseline_py_file.load_adj(DCRNN_ADJ, ADJTYPE)
#         supports = [torch.tensor(i).to(device) for i in adj_mx] if args.adj else None
#         model = baseline_py_file.gwn3(device, quantiles=quantiles, num_nodes=N_NODE, in_dim=CHANNEL, supports=None,
#                                       layers=2).to(device)
#     if args.model == 'gwn4':
#         adj_mx = baseline_py_file.load_adj(DCRNN_ADJ, ADJTYPE)
#         supports = [torch.tensor(i).to(device) for i in adj_mx] if args.adj else None
#         model = baseline_py_file.gwn4(device, quantiles=quantiles, num_nodes=N_NODE, in_dim=CHANNEL, supports=None,
#                                       layers=2).to(device)
    if args.model == 'dcrnn00':
#         adj_mx = baseline_py_file.load_adj(DCRNN_ADJ, ADJTYPE)
        adj_mx = baseline_py_file.load_pickle('./data/PEMS04/adj_PEMS04.pkl')
        supports=[]
        supports.append(baseline_py_file.calculate_random_walk_matrix(adj_mx).T)
        supports.append(baseline_py_file.calculate_random_walk_matrix(adj_mx.T).T)    
        supports = [torch.tensor(i).to(device) for i in supports]
        model = baseline_py_file.dcrnn00(device, adj_mx=supports, input_dim=CHANNEL, output_dim=3, seq_len=TIMESTEP_OUT, horizon=TIMESTEP_OUT, use_curriculum_learning=False, num_nodes=N_NODE, max_diffusion_step=2, cl_decay_steps=args.cl_size, filter_type='dual_random_walk', num_rnn_layers=2, rnn_units=64).to(device)
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
    if args.model == 'stgcn0':
        ks, kt, bs, T, n, p = 3, 3, [[CHANNEL, 16, 64], [64, 16, 64]], TIMESTEP_IN, N_NODE, 0
        A = pd.read_csv(STGCN_ADJ).values
        W = baseline_py_file.weight_matrix(A)
        L = baseline_py_file.scaled_laplacian(W)
        Lk = baseline_py_file.cheb_poly(L, ks)
        Lk = torch.Tensor(Lk.astype(np.float32)).to(device)
        model = baseline_py_file.stgcn0(ks, kt, bs, T, n, Lk, p, quantiles).to(device)
        ###############################################

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

    ## initial the model parameters ###
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)
    return model


def predictModel(name, model, data_iter):
    YS_truth = []
    YS_pred = []
    model.eval()
    with torch.no_grad():
        for x, y in data_iter:
            x = x[:, :, :, [0, -1]] if args.addtime else x[:, :, :, 0:1]
            y = y[:, :, :, 0:1]
            if name[:6] == 'stgcn1' :
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
                ypred = model(x)  # [b,t,n,q]      
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
    if point_prediction:
        val_rmse, val_mae, val_mape, val_smape, val_rse = lib.Metrics.point_evaluate(val_y_truth, val_y_pred)
        val_quantiles = []
    elif not with_point:
        val_rmse, val_mae, val_mape, val_smape, val_rse, val_quantiles = lib.Metrics.evaluate(val_y_truth, val_y_pred,
                                                                                              quantiles)  # [T]
    else:
        val_rmse, val_mae, val_mape, val_smape, val_rse, val_quantiles = lib.Metrics.evaluate_separates(val_y_truth,
                                                                                                        val_y_pred,
                                                                                                        quantiles)  # [T]
    test_y_truth, test_y_pred = predictModel(name, model, test_iter)
    test_y_pred = lib.Utils.z_inverse(test_y_pred, x_stats['mean'], x_stats['std'])
    # test_y_truth = lib.Utils.z_inverse(test_y_truth, x_stats['mean'], x_stats['std'])
    if point_prediction:
        test_rmse, test_mae, test_mape, test_smape, test_rse = lib.Metrics.point_evaluate(test_y_truth, test_y_pred)
        test_quantiles = []
    elif not with_point:
        test_rmse, test_mae, test_mape, test_smape, test_rse, test_quantiles = lib.Metrics.evaluate(test_y_truth,
                                                                                                    test_y_pred,
                                                                                                    quantiles)  # [T]
    else:
        test_rmse, test_mae, test_mape, test_smape, test_rse, test_quantiles = lib.Metrics.evaluate_separates(
            test_y_truth,
            test_y_pred,
            quantiles)  # [T]
    if save:
        np.save(single_version_PATH + '/' + args.data + '_' + MODELNAME + '_prediction_v'+args.version+'.npy', test_y_pred)
        np.save(single_version_PATH + '/' + args.data + '_' + MODELNAME + '_groundtruth_v'+args.version+'.npy', test_y_pred)
    return val_rmse, val_mae, val_mape, val_smape, val_rse, val_quantiles, test_rmse, test_mae, test_mape, test_smape, test_rse, test_quantiles

def trainModel(name, device, data, x_stats):
    mode = 'Train'
    print('Model Training Started ...', time.ctime())
    print('TIMESTEP_IN, TIMESTEP_OUT', TIMESTEP_IN, TIMESTEP_OUT)
    model = getModel(name, device)
    #     summary(model, (TIMESTEP_IN, N_NODE, CHANNEL), device=device)
    train_iter = torch_data_loader(device, data, data_type='train', shuffle=True)
    val_iter = torch_data_loader(device, data, data_type='val', shuffle=True)
    test_iter = torch_data_loader(device, data, data_type='test', shuffle=False)
    torch_mean = torch.Tensor(x_stats['mean'].reshape((1, 1, -1, 1))).to(device)
    torch_std = torch.Tensor(x_stats['std'].reshape((1, 1, -1, 1))).to(device)
    min_val_loss = np.inf
    wait = 0
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
        criterion = lib.Utils.QuantileLoss(quantiles)
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
        optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.0001, lr=LEARNING_RATE)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,gamma=0.1, milestones=[30], verbose=False)   #  milestones=[40], 
#     if args.model =='scinet':
#         optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=1e-5)
#         scheduler= torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.5)
        if args.sche:
            scheduler= torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 25, 35, 45], gamma=0.1)
    iter_count, cl_length = 1, 1
    batches_seen = 0
    if bool(args.cl):
        print(f"CL target length = %d" % cl_length)       
    for epoch in range(EPOCH):  # EPOCH
        starttime = datetime.now()
        loss_sum, n = 0.0, 0
        model.train()
        for x, y in train_iter:
            optimizer.zero_grad()
            x = x[:, :, :, [0, -1]] if args.addtime else x[:, :, :, 0:1]
            if name == 'stgcn1':
                y = y[:, 0:1, :, 0:1]
            else:
                y = y[:, :, :, 0:1]
            label = (y.clone()-torch_mean)/torch_std
            ypred = model(x, label, batches_seen)  # [b,t,n,q]
            ypred = ypred * torch_std + torch_mean
            if epoch < args.warm_epoch:
                loss = criterion(ypred, y)
            else:
                if args.cl:
                    if iter_count % args.cl_size == 0 and cl_length < TIMESTEP_OUT:
                        cl_length += 1
                        print(f"CL target length = %d" % cl_length)
                    if 'combined' in LOSS:
                        loss = criterion(ypred[:, :cl_length, :, :-1], y[:, :cl_length, :, :])
                        # loss = criterion(ypred[:, :cl_length, :, -1:], y[:, :cl_length, :, :])
                    else:
                        loss = criterion(ypred[:, :cl_length, :, :], y[:, :cl_length, :, :])
                    iter_count += 1
                else:
                    if 'combined' in LOSS:
                        loss = criterion(ypred[:, :, :, :-1], y)
                        # loss = criterion(ypred[:, :, :, -1:], y)
                    else:
                        loss = criterion(ypred, y)
            if batches_seen == 0:
                # this is a workaround to accommodate dynamically registered parameters in DCGRUCell
                optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, eps=1e-8)
                scheduler= torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 25, 35, 45], gamma=0.1)
            batches_seen += 1    
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * y.shape[0]
            n += y.shape[0]
#         if args.model=='scinet':
#             if (epoch+1) % 5 == 0:
#                 scheduler.step()
#         else:
        if args.sche:
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            print('Learning rate :', current_lr)

        if epoch % PRINT_EPOCH == 0:
            train_loss = loss_sum / n
            # 原版STGCN的点预测
            if args.model == 'stgcn0':
                val_rmse, val_mae, val_mape, val_smape, val_rse, val_quantiles, \
                test_rmse, test_mae, test_mape, test_smape, test_rse, test_quantiles = model_inference(name, model,
                                                                                                       val_iter,
                                                                                                       test_iter,
                                                                                                       x_stats,
                                                                                                       save=False,
                                                                                                       point_prediction=True)
            else:
                val_rmse, val_mae, val_mape, val_smape, val_rse, val_quantiles, \
                test_rmse, test_mae, test_mape, test_smape, test_rse, test_quantiles = model_inference(name, model,
                                                                                                       val_iter,
                                                                                                       test_iter,
                                                                                                       x_stats,
                                                                                                       save=False,
                                                                                                       with_point=False)
            print(f'Epoch {epoch}: ')
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

            with open(single_version_PATH + '/' + name + '_log.txt', 'a') as f:
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
            if args.cl:
                if cl_length == 12:
                    if np.mean(val_loss) < min_val_loss:
                        wait = 0
                        min_val_loss = np.mean(val_loss)
                        best_epoch=epoch
                        torch.save(model.state_dict(), single_version_PATH + '/' + name + '.pt')
                    else:
                        wait += 1
                    if wait == PATIENCE:
                        print('Early stopping at epoch: %d' % epoch)
                        break
            else:
                if np.mean(val_loss) < min_val_loss:
                    wait = 0
                    min_val_loss = np.mean(val_loss)
                    best_epoch=epoch
                    torch.save(model.state_dict(), single_version_PATH + '/' + name + '.pt')
                else:
                    wait += 1
                if wait == PATIENCE:
                    print('Early stopping at epoch: %d' % epoch)
                    break                
        endtime = datetime.now()
        epoch_time = (endtime - starttime).seconds
        print("epoch", epoch, "time used:", epoch_time, " seconds ", "train loss:", np.around(train_loss, 3),
              "val loss:", np.around(np.mean(val_loss), 6), "test loss:", np.around(np.mean(test_loss), 6))
        with open(single_version_PATH + '/' + name + '_log.txt', 'a') as f:
            f.write("%s, %d, %s, %d, %s, %s, %.3f, %s, %.3f, %s, %.3f\n" % (
                "epoch", epoch, "time used", epoch_time, "seconds", "train loss", train_loss, "validation loss:",
                np.mean(val_loss), "test loss:", np.mean(test_loss)))

    model.load_state_dict(torch.load(single_version_PATH + '/' + name + '.pt'))
    val_rmse, val_mae, val_mape, val_smape, val_rse, val_quantiles, \
    test_rmse, test_mae, test_mape, test_smape, test_rse, test_quantiles = model_inference(name, model, val_iter,
                                                                                           test_iter, x_stats,
                                                                                           save=False,
                                                                                           with_point=with_point)
    
    
    print('Model ', name, ' Best Results:')
    print(f'Epoch {best_epoch}: ')
    head = "%-8s %-8s %-8s %-8s %-8s %-8s %-8s %-8s %-8s %-8s %-8s %-8s %-8s %-8s %-8s %-8s %-8s %-8s"
    row = "%-8.3f %-8.3f %-8.3f %-8.3f %-8.3f %-8.3f %-8.3f %-8.3f %-8.3f %-8.3f %-8.3f %-8.3f %-8.3f %-8.3f %-8.3f %-8.3f %-8.3f %-8.3f"
    print(head % ("=", "=", "Horizon", "3", "=", "=", "=", "=", "Horizon", "6", "=", "=", "=", "=", "Horizon", "12", "=",  "="))
    print(head % ("RMSE", "MAE", "MAPE", "Q10", "Q50", "Q90","RMSE", "MAE", "MAPE", "Q10", "Q50", "Q90","RMSE", "MAE", "MAPE", "Q10", "Q50", "Q90"))
    print(row % (test_rmse[2], test_mae[2], test_mape[2], test_quantiles[2, 0], test_quantiles[2, 1], test_quantiles[2, 2],test_rmse[5], test_mae[5], test_mape[5], test_quantiles[5, 0], test_quantiles[5, 1], test_quantiles[5, 2], test_rmse[11], test_mae[11], test_mape[11], test_quantiles[11, 0], test_quantiles[11, 1],
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
    model = getModel(name, device)
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
        data, timestamp = data_preprocess(args.data)
        data, x_stats = get_inputdata(data, timestamp, which_train, ifaddtime=args.addtime)
        val_iter = torch_data_loader(device, data, data_type='val', shuffle=True)
        test_iter = torch_data_loader(device, data, data_type='test', shuffle=False)
        print('*' * 40)
        print('*' * 40)
        print('Under Train Strategy --- ', tr, ' ---:')
        for ind, v_ in enumerate(versions):
            print('--- version ', v_, ' evaluation start ---')
            multi_test_PATH = \
                "./save/{}_{}_in{}_out{}_addtime{}_adj{}_lr{}_hc{}_train{}_val{}_test{}_seed{}_loss{}_version{}_note:{}/{}.pt".format(
                    DATANAME, args.model, args.instep, args.outstep, args.addtime, args.adj, LEARNING_RATE, args.hc,
                    which_train,
                    VAL, TEST, SEED, LOSS, v_, args.note, args.model)
            if os.path.isfile(multi_test_PATH):
                if args.model=='dcrnn00':
                    print('set graph')
                    model.eval()
                    for _, (x, y) in enumerate(val_iter):
                        x = x[:, :, :, [0, -1]] if args.addtime else x[:, :, :, 0:1]
                        output = model(x)
                        break                
                model.load_state_dict(torch.load(multi_test_PATH, map_location=device))
                print("file exists: ", multi_test_PATH)
            else:
                print("file not exist", multi_test_PATH)
                break
            print('*' * 20)
            print(f'Version: {v_} Start Testing :')
            val_rmse, val_mae, val_mape, val_smape, val_rse, val_quantiles, \
            test_rmse, test_mae, test_mape, test_smape, test_rse, test_quantiles = model_inference(name, model,
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
    print(head % ("=", "=", "Horizon", "3", "=", "=", "=", "=", "Horizon", "6", "=", "=",  "=", "=", "Horizon", "12", "=", "="))
    print(head % ("RMSE", "MAE", "MAPE", "Q10", "Q50", "Q90","RMSE", "MAE", "MAPE", "Q10", "Q50", "Q90","RMSE", "MAE", "MAPE", "Q10", "Q50", "Q90"))
    print(row % (rmse[2], mae[2], mape[2], total_quantiles[2, 0], total_quantiles[2, 1], total_quantiles[2, 2],rmse[5], mae[5], mape[5], total_quantiles[5, 0], total_quantiles[5, 1], total_quantiles[5, 2], rmse[11], mae[11], mape[11], total_quantiles[11, 0], total_quantiles[11, 1],
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
    timestamp, data = data_preprocess(args.data)
    if args.mode == 'train':  # train and test in single version
        if not os.path.exists(single_version_PATH):
            os.makedirs(single_version_PATH)
        print(single_version_PATH, 'training started', time.ctime())
        model_path = './model/' + args.model + '.py'
        shutil.copy2(model_path, single_version_PATH)
        data, timestamp = data_preprocess(args.data)
        seq_data, x_stats = get_inputdata(data, timestamp, train=args.train, ifaddtime=args.addtime)
        #         data,timestamp=data_preprocess(args.data)
        #         seq_data, x_stats = get_inputdata(data, timestamp, train=args.train, ifaddtime=False)
        for key in seq_data.keys():
            print(key, ' : ', seq_data[key].shape)
        trainModel(MODELNAME, device, seq_data, x_stats)
    if args.mode == 'eval0':  # eval in sing`le version
        print('single version ', args.version, ' testing started', time.ctime())
        multi_version_test(MODELNAME, device, train=[args.train], versions=[args.version])  #
    if args.mode == 'eval':  # eval in multi version
        #         print(multi_version_PATH, 'multi version testing started', time.ctime())
        multi_version_test(MODELNAME, device, train=[args.train], versions=np.arange(0, 5))  #
    if args.mode == 'all':
        if not os.path.exists(multi_version_PATH):
            os.makedirs(multi_version_PATH)
        multi_version_test(MODELNAME, device, train=[0.7], versions=[int(i) for i in np.arange(0, 5)])  #


if __name__ == '__main__':
    main()
