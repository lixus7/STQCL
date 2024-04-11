import torch
from torch import nn

import pickle
import sys
import numpy as np
import scipy.sparse as sp
from torchsummary import summary

import torch

import numpy as np
import torch

def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian


def calculate_random_walk_matrix(adj_mx):
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1)).flatten()
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).astype(np.float32).todense()
    return random_walk_mx


def calculate_reverse_random_walk_matrix(adj_mx):
    return calculate_random_walk_matrix(np.transpose(adj_mx))


def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32)

class LayerParams:
    def __init__(self, rnn_network: torch.nn.Module, layer_type: str):
        self._rnn_network = rnn_network
        self._params_dict = {}
        self._biases_dict = {}
        self._type = layer_type

    def get_weights(self, shape):
        if shape not in self._params_dict:
            nn_param = torch.nn.Parameter(torch.empty(*shape))
            torch.nn.init.xavier_normal_(nn_param)
            self._params_dict[shape] = nn_param
            self._rnn_network.register_parameter('{}_weight_{}'.format(self._type, str(shape)),
                                                 nn_param)
        return self._params_dict[shape]

    def get_biases(self, length, bias_start=0.0):
        if length not in self._biases_dict:
            biases = torch.nn.Parameter(torch.empty(length))
            torch.nn.init.constant_(biases, bias_start)
            self._biases_dict[length] = biases
            self._rnn_network.register_parameter('{}_biases_{}'.format(self._type, str(length)),
                                                 biases)

        return self._biases_dict[length]


class DCGRUCell(torch.nn.Module):
    def __init__(self, device, num_units, adj_mx, max_diffusion_step, num_nodes, nonlinearity='tanh',
                 filter_type="dual_random_walk", use_gc_for_ru=True):
        """

        :param num_units:
        :param adj_mx:
        :param max_diffusion_step:
        :param num_nodes:
        :param nonlinearity:
        :param filter_type: "laplacian", "random_walk", "dual_random_walk".
        :param use_gc_for_ru: whether to use Graph convolution to calculate the reset and update gates.
        """

        super().__init__()
        self._activation = torch.tanh if nonlinearity == 'tanh' else torch.relu
        # support other nonlinearities up here?
        self._num_nodes = num_nodes
        self._num_units = num_units
        self._max_diffusion_step = max_diffusion_step
        self._supports = []
        self._use_gc_for_ru = use_gc_for_ru
        self.device=device
#         supports = []
#         if filter_type == "laplacian":
#             supports.append(calculate_scaled_laplacian(adj_mx, lambda_max=None))
#         elif filter_type == "random_walk":
#             supports.append(calculate_random_walk_matrix(adj_mx).T)
#         elif filter_type == "dual_random_walk":
#             supports.append(calculate_random_walk_matrix(adj_mx).T)
#             supports.append(calculate_random_walk_matrix(adj_mx.T).T)
#         else:
#             supports.append(calculate_scaled_laplacian(adj_mx))
#         supports = [torch.tensor(i).to(device) for i in supports]    
#         for support in supports:
#             self._supports.append(self._build_sparse_matrix(support,device))
        self._supports = adj_mx
        self._fc_params = LayerParams(self, 'fc')
        self._gconv_params = LayerParams(self, 'gconv')

    @staticmethod
    def _build_sparse_matrix(L,device):
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        # this is to ensure row-major ordering to equal torch.sparse.sparse_reorder(L)
        indices = indices[np.lexsort((indices[:, 0], indices[:, 1]))]
        L = torch.sparse_coo_tensor(indices.T, L.data, L.shape, device=device)
        return L

    def forward(self, inputs, hx):
        """Gated recurrent unit (GRU) with Graph Convolution.
        :param inputs: (B, num_nodes * input_dim)
        :param hx: (B, num_nodes * rnn_units)

        :return
        - Output: A `2-D` tensor with shape `(B, num_nodes * rnn_units)`.
        """
        output_size = 2 * self._num_units
        if self._use_gc_for_ru:
            fn = self._gconv
        else:
            fn = self._fc
        value = torch.sigmoid(fn(inputs, hx, output_size, bias_start=1.0))
        value = torch.reshape(value, (-1, self._num_nodes, output_size))
        r, u = torch.split(tensor=value, split_size_or_sections=self._num_units, dim=-1)
        r = torch.reshape(r, (-1, self._num_nodes * self._num_units))
        u = torch.reshape(u, (-1, self._num_nodes * self._num_units))

        c = self._gconv(inputs, r * hx, self._num_units)
        if self._activation is not None:
            c = self._activation(c)

        new_state = u * hx + (1.0 - u) * c
        return new_state

    @staticmethod
    def _concat(x, x_):
        x_ = x_.unsqueeze(0)
        return torch.cat([x, x_], dim=0)

    def _fc(self, inputs, state, output_size, bias_start=0.0):
        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size * self._num_nodes, -1))
        state = torch.reshape(state, (batch_size * self._num_nodes, -1))
        inputs_and_state = torch.cat([inputs, state], dim=-1)
        input_size = inputs_and_state.shape[-1]
        weights = self._fc_params.get_weights((input_size, output_size))
        value = torch.sigmoid(torch.matmul(inputs_and_state, weights))
        biases = self._fc_params.get_biases(output_size, bias_start)
        value += biases
        return value

    def _gconv(self, inputs, state, output_size, bias_start=0.0):
        # Reshape input and state to (batch_size, num_nodes, input_dim/state_dim)
        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size, self._num_nodes, -1))
        state = torch.reshape(state, (batch_size, self._num_nodes, -1))
        inputs_and_state = torch.cat([inputs, state], dim=2)
        input_size = inputs_and_state.size(2)

        x = inputs_and_state
        x0 = x.permute(1, 2, 0)  # (num_nodes, total_arg_size, batch_size)
        x0 = torch.reshape(x0, shape=[self._num_nodes, input_size * batch_size])
        x = torch.unsqueeze(x0, 0)

        if self._max_diffusion_step == 0:
            pass
        else:
            for support in self._supports:
                x1 = torch.sparse.mm(support, x0)
                x = self._concat(x, x1)

                for k in range(2, self._max_diffusion_step + 1):
                    x2 = 2 * torch.sparse.mm(support, x1) - x0
                    x = self._concat(x, x2)
                    x1, x0 = x2, x1

        num_matrices = len(self._supports) * self._max_diffusion_step + 1  # Adds for x itself.
        x = torch.reshape(x, shape=[num_matrices, self._num_nodes, input_size, batch_size])
        x = x.permute(3, 1, 2, 0)  # (batch_size, num_nodes, input_size, order)
        x = torch.reshape(x, shape=[batch_size * self._num_nodes, input_size * num_matrices])

        weights = self._gconv_params.get_weights((input_size * num_matrices, output_size)).to(self.device)
        x = torch.matmul(x, weights)  # (batch_size * self._num_nodes, output_size)

        biases = self._gconv_params.get_biases(output_size, bias_start).to(self.device)
        x += biases
        # Reshape res back to 2D: (batch_size, num_node, state_dim) -> (batch_size, num_node * state_dim)
        return torch.reshape(x, [batch_size, self._num_nodes * output_size])
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)




class EncoderModel(nn.Module):
    def __init__(self, device, input_dim, seq_len, adj_mx, num_nodes, max_diffusion_step,cl_decay_steps, filter_type, num_rnn_layers, rnn_units):
        nn.Module.__init__(self)
        self.input_dim = input_dim
        self.device = device
        self.seq_len = seq_len  # for the encoder
        self.hidden_state_size = num_nodes * rnn_units
        self.num_rnn_layers = num_rnn_layers
        self.dcgru_layers = nn.ModuleList(
            [DCGRUCell(device, rnn_units, adj_mx, max_diffusion_step, num_nodes,
                       filter_type=filter_type) for _ in range(self.num_rnn_layers)])

    def forward(self, inputs, hidden_state=None):
        """
        Encoder forward pass.

        :param inputs: shape (batch_size, self.num_nodes * self.input_dim)
        :param hidden_state: (num_layers, batch_size, self.hidden_state_size)
               optional, zeros if not provided
        :return: output: # shape (batch_size, self.hidden_state_size)
                 hidden_state # shape (num_layers, batch_size, self.hidden_state_size)
                 (lower indices mean lower layers)
        """
        batch_size, _ = inputs.size()
        if hidden_state is None:
            hidden_state = torch.zeros((self.num_rnn_layers, batch_size, self.hidden_state_size),
                                       device=self.device)
        hidden_states = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num])
            hidden_states.append(next_hidden_state)
            output = next_hidden_state

        return output, torch.stack(hidden_states)  # runs in O(num_layers) so not too slow


class DecoderModel(nn.Module):
    def __init__(self, device, output_dim, horizon, adj_mx, num_nodes, max_diffusion_step,cl_decay_steps, filter_type, num_rnn_layers, rnn_units):
        # super().__init__(is_training, adj_mx, **model_kwargs)
        nn.Module.__init__(self)
        self.output_dim = output_dim
        self.horizon = horizon  # for the decoder
        self.projection_layer = nn.Linear(rnn_units, output_dim)
        self.rnn_units = rnn_units
        self.num_nodes = num_nodes
        self.device = device
        self.dcgru_layers = nn.ModuleList(
            [DCGRUCell(device, rnn_units, adj_mx, max_diffusion_step, num_nodes,filter_type=filter_type) for _ in range(num_rnn_layers)])

    def forward(self, inputs, hidden_state=None):
        """
        Decoder forward pass.

        :param inputs: shape (batch_size, self.num_nodes * self.output_dim)
        :param hidden_state: (num_layers, batch_size, self.hidden_state_size)
               optional, zeros if not provided
        :return: output: # shape (batch_size, self.num_nodes * self.output_dim)
                 hidden_state # shape (num_layers, batch_size, self.hidden_state_size)
                 (lower indices mean lower layers)
        """
        hidden_states = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num])
            hidden_states.append(next_hidden_state)
            output = next_hidden_state

        projected = self.projection_layer(output.view(-1, self.rnn_units))
        output = projected.view(-1, self.num_nodes * self.output_dim)

        return output, torch.stack(hidden_states)


class dcrnn0(nn.Module):
    def __init__(self, device, adj_mx, input_dim, output_dim, seq_len, horizon, use_curriculum_learning, num_nodes, max_diffusion_step, cl_decay_steps, filter_type, num_rnn_layers, rnn_units):
        super().__init__()
        self.encoder_model = EncoderModel(device, input_dim, seq_len, adj_mx, num_nodes, max_diffusion_step,cl_decay_steps, filter_type, num_rnn_layers, rnn_units)
        self.decoder_model = DecoderModel(device, output_dim, horizon, adj_mx, num_nodes, max_diffusion_step,cl_decay_steps, filter_type, num_rnn_layers, rnn_units)
        self.cl_decay_steps = cl_decay_steps
        self.use_curriculum_learning = use_curriculum_learning
        self.num_nodes = num_nodes
        self.device =device
        

    def _compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (
                self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def encoder(self, inputs):
        """
        encoder forward pass on t time steps
        :param inputs: shape (seq_len, batch_size, num_sensor * input_dim)
        :return: encoder_hidden_state: (num_layers, batch_size, self.hidden_state_size)
        """
        encoder_hidden_state = None
        for t in range(self.encoder_model.seq_len):
            _, encoder_hidden_state = self.encoder_model(inputs[t], encoder_hidden_state)

        return encoder_hidden_state

    def decoder(self, encoder_hidden_state, labels=None, batches_seen=None):
        """
        Decoder forward pass
        :param encoder_hidden_state: (num_layers, batch_size, self.hidden_state_size)
        :param labels: (self.horizon, batch_size, self.num_nodes * self.output_dim) [optional, not exist for inference]
        :param batches_seen: global step [optional, not exist for inference]
        :return: output: (self.horizon, batch_size, self.num_nodes * self.output_dim)
        """
        batch_size = encoder_hidden_state.size(1)
        go_symbol = torch.zeros((batch_size, self.num_nodes * self.decoder_model.output_dim),
                                device=self.device)
        decoder_hidden_state = encoder_hidden_state
        decoder_input = go_symbol

        outputs = []

        for t in range(self.decoder_model.horizon):
            decoder_output, decoder_hidden_state = self.decoder_model(decoder_input,
                                                                      decoder_hidden_state)
            decoder_input = decoder_output
            outputs.append(decoder_output)
            if self.training and self.use_curriculum_learning:
                c = np.random.uniform(0, 1)
                if c < self._compute_sampling_threshold(batches_seen):
                    decoder_input = labels[t]
        outputs = torch.stack(outputs)
        return outputs

    def forward(self, inputs, labels=None, batches_seen=None):
        """
        seq2seq forward pass
        :param inputs: shape (seq_len, batch_size, num_sensor * input_dim)
        :param labels: shape (horizon, batch_size, num_sensor * output)
        :param batches_seen: batches seen till now
        :return: output: (self.horizon, batch_size, self.num_nodes * self.output_dim)
        """
        # reshape data
        batch_size, length, num_nodes, channels = inputs.shape
        inputs = inputs.reshape(batch_size, length, num_nodes * channels)      # [B, L, N*C]
        inputs = inputs.transpose(0, 1)         # [L, B, N*C]
        if labels is not None:
            labels = labels[..., [0]]     # teacher forcing only use the first dimension.
            batch_size, length, num_nodes, channels = labels.shape
            labels = labels.reshape(batch_size, length, num_nodes * channels)      # [B, L, N*C]
            labels = labels.transpose(0, 1)         # [L, B, N*C]       
        encoder_hidden_state = self.encoder(inputs)
        # print("Encoder complete, starting decoder")
        outputs = self.decoder(encoder_hidden_state, labels, batches_seen=batches_seen)
        #Q printdebug("Decoder complete")
#         if batches_seen == 0:
#             self._logger.info(
#                 "Total trainable parameters {}".format(count_parameters(self))
#             )
        # reshape to B, L, N, C¡¢
        # print('outputs shape is ',outputs.shape)
        L, B, _ = outputs.shape
        outputs = outputs.transpose(0, 1)  # [B, L, N*C_out]
        outputs = outputs.view(B, L, self.decoder_model.num_nodes,
                               self.decoder_model.output_dim)
        # print('out shape is:',outputs.shape)
        return outputs
import pickle
def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data
def main():
    CHANNEL = 1
    N_NODE = 307
    TIMESTEP_IN = 12
    TIMESTEP_OUT = 12
    QUANTILES = [0.1,0.5,0.9]
    ADJPATH = '../data/PEMS04/adj_PEMS04.pkl'
    ADJTYPE = 'doubletransition'
    GPU = sys.argv[-1] if len(sys.argv) == 2 else '7'
    device = torch.device("cuda:{}".format(GPU)) if torch.cuda.is_available() else torch.device("cpu")
    # adj_mx = load_adj(ADJPATH, ADJTYPE)
#     sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(ADJPATH)
    adj_mx = load_pickle('../data/PEMS04/adj_PEMS04.pkl')
    model = dcrnn0(device, adj_mx, input_dim=1, output_dim=1, seq_len=12, horizon=12, use_curriculum_learning=False, num_nodes=307, max_diffusion_step=2, cl_decay_steps=2000, filter_type='dual_random_walk', num_rnn_layers=2, rnn_units=64).to(device)
    x=torch.randn((8,12,307,1)).to(device)
    y=torch.randn((8,12,307,1)).to(device)
    out = model(x,y,0)
    # summary(model, (TIMESTEP_IN, N_NODE, CHANNEL), device=device)

if __name__ == '__main__':
    main()