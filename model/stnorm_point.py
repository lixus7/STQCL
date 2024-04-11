import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter
import sys


class SNorm(nn.Module):
    def __init__(self,  channels):
        super(SNorm, self).__init__()
        self.beta = nn.Parameter(torch.zeros(channels))
        self.gamma = nn.Parameter(torch.ones(channels))

    def forward(self, x):
        x_norm = (x - x.mean(2, keepdims=True)) / (x.var(2, keepdims=True, unbiased=True) + 0.00001) ** 0.5

        out = x_norm * self.gamma.view(1, -1, 1, 1) + self.beta.view(1, -1, 1, 1)
        return out


class TNorm(nn.Module):
    def __init__(self,  num_nodes, channels, track_running_stats=True, momentum=0.1):
        super(TNorm, self).__init__()
        self.track_running_stats = track_running_stats
        self.beta = nn.Parameter(torch.zeros(1, channels, num_nodes, 1))
        self.gamma = nn.Parameter(torch.ones(1, channels, num_nodes, 1))
        self.register_buffer('running_mean', torch.zeros(1, channels, num_nodes, 1))
        self.register_buffer('running_var', torch.ones(1, channels, num_nodes, 1))
        self.momentum = momentum

    def forward(self, x):
        if self.track_running_stats:
            mean = x.mean((0, 3), keepdims=True)
            var = x.var((0, 3), keepdims=True, unbiased=False)
            if self.training:
                n = x.shape[3] * x.shape[0]
                with torch.no_grad():
                    self.running_mean = self.momentum * mean + (1 - self.momentum) * self.running_mean
                    self.running_var = self.momentum * var * n / (n - 1) + (1 - self.momentum) * self.running_var
            else:
                mean = self.running_mean
                var = self.running_var
        else:
            mean = x.mean((3), keepdims=True)
            var = x.var((3), keepdims=True, unbiased=True)
        x_norm = (x - mean) / (var + 0.00001) ** 0.5
        out = x_norm * self.gamma + self.beta
        return out


class stnorm_point(nn.Module):
    def __init__(self, device, quantiles, num_nodes, tnorm_bool=False, snorm_bool=False, in_dim=1,out_dim=12, channels=16,kernel_size=2,blocks=4,layers=2):
        super(stnorm_point, self).__init__()
        self.blocks = blocks
        self.layers = layers
        self.quantiles = quantiles
        self.snorm_bool = snorm_bool
        self.tnorm_bool = tnorm_bool
        self.num_nodes = num_nodes
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()

        if self.snorm_bool:
            self.sn = nn.ModuleList()
        if self.tnorm_bool:
            self.tn = nn.ModuleList()
        num = int(self.tnorm_bool) + int(self.snorm_bool) + 1

        self.mlps = nn.ModuleList()
        self.gconv = nn.ModuleList()
        self.cross_product = nn.ModuleList()

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=channels,
                                    kernel_size=(1,1))

        receptive_field = 1
        self.dropout = nn.Dropout(0.2)

        self.dilation = []

        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilated convolutions
                self.dilation.append(new_dilation)
                if self.tnorm_bool:
                    self.tn.append(TNorm(num_nodes, channels))
                if self.snorm_bool:
                    self.sn.append(SNorm(channels))
                self.filter_convs.append(nn.Conv2d(in_channels=num * channels,
                                                   out_channels=channels,
                                                   kernel_size=(1,kernel_size),dilation=new_dilation))

                self.gate_convs.append(nn.Conv2d(in_channels=num * channels,
                                                 out_channels=channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv2d(in_channels=channels,
                                                     out_channels=channels,
                                                     kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv2d(in_channels=channels,
                                                 out_channels=channels,
                                                 kernel_size=(1, 1)))
                new_dilation *=2
                receptive_field += additional_scope
                additional_scope *= 2



        self.end_conv_1 = nn.Conv2d(in_channels=channels,
                                  out_channels=channels,
                                  kernel_size=(1,1),
                                  bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=channels,
                                    out_channels=out_dim,
                                    kernel_size=(1,1),
                                    bias=True)

        self.receptive_field = receptive_field

    def forward(self, input):
        # input shape is : [B,T,N,C]
        # , mean, std, y=None, p=None, c_epoch=None):
        input = input.permute(0, 3, 2, 1)
        in_len = input.size(3)
        if in_len<self.receptive_field:
            x = nn.functional.pad(input,(self.receptive_field-in_len,0,0,0))
        else:
            x = input
        x = self.start_conv(x)
        skip = 0


        # WaveNet layers
        for i in range(self.blocks * self.layers):
            residual = x
            x_list = []
            x_list.append(x)
            b, c, n, t = x.shape
            if self.tnorm_bool:
                x_tnorm = self.tn[i](x)
                x_list.append(x_tnorm)
            if self.snorm_bool:
                x_snorm = self.sn[i](x)
                x_list.append(x_snorm)
            # dilated convolution
            x = torch.cat(x_list, dim=1)
            filter = self.filter_convs[i](x)
            b, c, n, t = filter.shape
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](x)
            gate = torch.sigmoid(gate)
            x = filter * gate
            # parametrized skip connection
            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :,  -s.size(3):]
            except:
                skip = 0
            skip = s + skip

            x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]

        x = F.relu(skip)
        rep = F.relu(self.end_conv_1(x))
        out = self.end_conv_2(rep)
        # print('out shape is :',out.shape)
        out = out[:, :, :, 0].permute(0, 2, 1).reshape(-1, self.num_nodes, out.shape[1], 1).permute(0,2,1,3)
        # print('out shape is :',out.shape)
        return out

    def load_my_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if isinstance(param, Parameter):
                param = param.data
            try:
                own_state[name].copy_(param)
            except:
                print(name)
                print(param.shape)

def print_params(model_name, model):
    param_count=0
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_count += param.numel()
    print(f'{model_name}, {param_count} trainable parameters in total.')
    return    

def main():
    import sys
    from torchsummary import summary
    N_NODE = 228
    TIMESTEP_IN = 16
    TIMESTEP_OUT = 3
    CHANNEL = 1
    GPU = sys.argv[-1] if len(sys.argv) == 2 else '3'
    device = torch.device("cuda:{}".format(GPU)) if torch.cuda.is_available() else torch.device("cpu")
    model =  stnorm_point(device, N_NODE, tnorm_bool=True, snorm_bool=True, in_dim=1,out_dim=TIMESTEP_OUT, channels=32, kernel_size=2, blocks=1, layers=4).to(device)
#     model = AGCRN(num_nodes=N_NODE, input_dim=CHANNEL, output_dim=CHANNEL, horizon=TIMESTEP_OUT).to(device)
    summary(model, (TIMESTEP_IN, N_NODE, CHANNEL), device=device)
    print_params('stnorm_point',model)    
if __name__ == '__main__':
    main()                
                