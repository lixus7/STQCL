import torch
from torch import nn
import torch.nn.functional as F 
from torch.optim import Adam
import sys
import numpy as np

class Decoder(nn.Module):

    def __init__(
        self, 
        input_size, 
        output_horizon,
        encoder_hidden_size, 
        decoder_hidden_size, 
        output_size):
        super(Decoder, self).__init__()
        self.global_mlp = nn.Linear(output_horizon * (encoder_hidden_size + input_size), \
                (output_horizon+1) * decoder_hidden_size)
        self.local_mlp = nn.Linear(decoder_hidden_size * 2 + input_size, output_size)
        self.decoder_hidden_size = decoder_hidden_size
    
    def forward(self, ht, xf):
        '''
        Args:
        ht (tensor): (1, hidden_size)
        xf (tensor): (output_horizon, num_features)
        '''
        num_ts, output_horizon, num_features = xf.size()
        num_ts, hidden_size = ht.size()
        ht = ht.unsqueeze(1)
        ht = ht.expand(num_ts, output_horizon, hidden_size)
        # inp = (xf + ht).view(batch_size, -1) # batch_size, hidden_size, output_horizon
        inp = torch.cat([xf, ht], dim=2).view(num_ts, -1)
        contexts = self.global_mlp(inp)
        contexts = contexts.view(num_ts, output_horizon+1, self.decoder_hidden_size)
        ca = contexts[:, -1, :].view(num_ts, -1)
        C = contexts[:, :-1, :]
        C = F.relu(C)
        y = []
        for i in range(output_horizon):
            ci = C[:, i, :].view(num_ts, -1)
            xfi = xf[:, i, :].view(num_ts, -1)
            inp = torch.cat([xfi, ci, ca], dim=1)
            out = self.local_mlp(inp) # num_ts, num_quantiles
            y.append(out.unsqueeze(1))
        y = torch.cat(y, dim=1) # batch_size, output_horizon, quantiles
        return y 


class MQRNN(nn.Module):

    def __init__(
        self, 
        output_horizon, 
        num_quantiles, 
        input_size, 
        embedding_size=10,
        encoder_hidden_size=50, 
        encoder_n_layers=1,
        decoder_hidden_size=20
        ):
        '''
        Args:
        output_horizon (int): output horizons to output in prediction
        num_quantiles (int): number of quantiles interests, e.g. 0.25, 0.5, 0.75
        input_size (int): feature size
        embedding_size (int): embedding size
        encoder_hidden_size (int): hidden size in encoder
        encoder_n_layers (int): encoder number of layers
        decoder_hidden_size (int): hidden size in decoder
        '''
        super(MQRNN, self).__init__()
        self.output_horizon = output_horizon
        self.encoder_hidden_size = encoder_hidden_size
        self.input_embed = nn.Linear(1, embedding_size) # time series embedding
        self.encoder = nn.LSTM(input_size + embedding_size, encoder_hidden_size, \
                    encoder_n_layers, bias=True, batch_first=True)
        self.decoder = Decoder(input_size, output_horizon, encoder_hidden_size,\
                    decoder_hidden_size, num_quantiles)
    
    def forward(self, X, y, Xf):
        '''
        Args:
        X (tensor like): shape (num_time_series, num_periods, num_features)  [batch*node,time,channel]
        y (tensor like): shape (num_time_series, num_periods) [batch*node,time]
        Xf (tensor like): shape (num_time_series, seq_len, num_features)  [batch*node,pred_length,channel]
        '''
#         if isinstance(X, type(np.empty(2))):
#             X = torch.from_numpy(X).float()
#             y = torch.from_numpy(y).float()
#             Xf = torch.from_numpy(Xf).float()
        num_ts, num_periods, num_features = X.shape
        y = torch.unsqueeze(y,2)
        y = self.input_embed(y)
        x = torch.cat([X, y], dim=2)
        # x = x.unsqueeze(0) # batch, seq_len, embed + num_features
        _, (h, c) = self.encoder(x)
        ht = h[-1, :, :]
        # global mlp
        ht = F.relu(ht)
        ypred = self.decoder(ht, Xf)
        # print(ypred.shape) # (batch*node, output_horizon, num_quantiles)
        return ypred
    
'''
MQRNN Metric Utils:
'''
def RSE(ypred, ytrue):
    '''
    Args:
    ytrue (batch_size, output_horizon)
    ypred (batch_size, output_horizon, num_quantiles)
    return rse(output_horizon)
    '''    
    ypred = ypred[:,:,1]
    rse = []
    for i in range(ytrue.shape[1]):
        single_horizon_rse = np.sqrt(np.square(ypred[:,i] - ytrue[:,i]).sum()) / \
            np.sqrt(np.square(ytrue[:,i] - ytrue[:,i].mean()).sum())
        rse.append(single_horizon_rse)
    rse = np.array(rse)
    return rse

def quantile_loss(ytrue, ypred, qs):
    '''
    Quantile loss version 2
    Args:
    ytrue (batch_size, output_horizon)
    ypred (batch_size, output_horizon, num_quantiles)
    return L(float), out(batch_size, output_horizon,num_quantiles)
    '''
    out=[]
    for i, q in enumerate(qs):
        L = np.zeros_like(ytrue).astype(np.float64)
        yq = ypred[:, :, i]
        diff = yq - ytrue
        L += np.maximum(q * diff.astype(np.float64), (q - 1) * diff.astype(np.float64))
        out.append(L)
    quantile_out = np.array(out).transpose(1,2,0)
    return L.mean(), np.mean(quantile_out,axis=0)

def SMAPE(ytrue, ypred):
    ytrue = np.array(ytrue).ravel()
    ypred = np.array(ypred).ravel() + 1e-4
    mean_y = (ytrue + ypred) / 2.
    return np.mean(np.abs((ytrue - ypred) \
        / mean_y))

def MAPE(ytrue, ypred):
    ytrue = np.array(ytrue).ravel() + 1e-4
    ypred = np.array(ypred).ravel()
    return np.mean(np.abs((ytrue - ypred) \
        / ytrue))

    
def main():
    GPU = sys.argv[-1] if len(sys.argv) == 2 else '3'
    device = torch.device("cuda:{}".format(GPU)) if torch.cuda.is_available() else torch.device("cpu")
    quantiles = [0.1,0.5,0.9]
    batch = 1000
    X_train_tensor = torch.randn((batch,168,2)).to(device)  # [batch*node,input_time,channel]
    y_train_tensor = torch.randn((batch,168)).to(device) # [batch*node,input_time]
    xf = torch.randn((batch,12,2)).to(device)         # [batch*node,pred_length,channel]
    model = MQRNN(
        output_horizon = 12, 
        num_quantiles = len(quantiles), 
        input_size = 2
        ).to(device)
#     summary(model, (TIMESTEP_IN, N_NODE, CHANNEL), device=device)
    ypred = model(X_train_tensor,y_train_tensor,xf)
    
    
    
if __name__ == '__main__':
    main()
