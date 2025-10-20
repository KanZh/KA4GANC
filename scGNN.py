import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optm
from torch.nn import CosineSimilarity
from torch_geometric.utils import *
from kan_conv import KANConv2DLayer
from FKAN import FKANLayer


class KA4GANC(nn.Module):
    def __init__(self, input_dim, hidden1_dim, hidden2_dim, hidden3_dim, output_dim, num_head1,
                 alpha, device):
        super(KA4GANC, self).__init__()
        self.input_dim = input_dim
        self.hidden1_dim = num_head1 * hidden1_dim
        self.hidden2_dim = hidden2_dim
        self.hidden3_dim = hidden3_dim

        self.num_head1 = num_head1
        self.device = device
        self.alpha = alpha

        self.ConvLayer1 = nn.ModuleList([
            Conv.KanGNN(self.input_dim, 128, self.hidden2_dim, 150, 1) for _ in range(1)
        ])
        self.add_module('KanGNN', self.ConvLayer1)
        self.ConvLayer12 = nn.ModuleList([
            Conv.KanGNN(self.hidden2_dim, 128, self.hidden3_dim, 150, 1) for _ in range(1)
        ])
        self.add_module('KanGNN', self.ConvLayer12)
        self.ConvLayer123 = nn.ModuleList([
            Conv.KanGNN(self.hidden3_dim, 128, 1, 150, 1) for _ in range(1)
        ])
        self.add_module('KanGNN', self.ConvLayer123)

        self.ConvLayer2 = Conv(
            in_channels=1,
            out_channels=64,
            kernel_size=(1),
            hidden_dim=self.hidden2_dim
        )
        self.add_module('ConvAttentionLayer', self.ConvLayer2)

        self.tf_linear1 = nn.Linear(hidden2_dim, hidden3_dim)
        self.target_linear1 = nn.Linear(hidden2_dim, hidden3_dim)

        self.tf_linear2 = nn.Linear(hidden3_dim, output_dim)
        self.target_linear2 = nn.Linear(hidden3_dim, output_dim)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.tf_linear1.weight, gain=1.414)
        nn.init.xavier_uniform_(self.target_linear1.weight, gain=1.414)
        nn.init.xavier_uniform_(self.tf_linear2.weight, gain=1.414)
        nn.init.xavier_uniform_(self.target_linear2.weight, gain=1.414)

    def encode(self, x, adj):
        x = torch.mean(torch.stack([layer(x, adj) for layer in self.ConvLayer1]), dim=0)
        x = F.leaky_relu(x)
        x = torch.mean(torch.stack([layer(x, adj) for layer in self.ConvLayer12]), dim=0)
        x = F.leaky_relu(x)
        x = torch.mean(torch.stack([layer(x, adj) for layer in self.ConvLayer123]), dim=0)
        x = F.leaky_relu(x)
        x = x.reshape((1, 1, x.shape[0], x.shape[1]))
        out = self.ConvLayer2(x, adj)

        return out

    def decode(self, tf_embed, target_embed):
        prob = torch.mul(tf_embed, target_embed)
        prob = torch.sum(prob, dim=1).view(-1, 1)

        return prob

    def forward(self, x, adj, train_sample):
        embed = self.encode(x, adj)

        tf_embed = self.tf_linear1(embed)
        tf_embed = F.dropout(tf_embed, p=0.01)
        tf_embed = self.tf_linear2(tf_embed)

        target_embed = self.target_linear1(embed)
        target_embed = F.dropout(target_embed, p=0.01)
        target_embed = self.target_linear2(target_embed)

        self.tf_ouput = tf_embed
        self.target_output = target_embed

        train_tf = tf_embed[train_sample[:, 0]]
        train_target = target_embed[train_sample[:, 1]]

        pred = self.decode(train_tf, train_target)

        return pred

    def get_embedding(self):
        return self.tf_ouput, self.target_output


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, hidden_dim, stride=1, padding=0, alpha=0.2, bias=True):
        super(Conv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.hidden_dim = hidden_dim
        self.alpha = alpha
        self.bias = bias

        self.a = nn.Parameter(torch.zeros(size=(2 * self.hidden_dim, 1)))
        self.conv = KANConv2DLayer(in_channels, out_channels, kernel_size, groups=1)

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(self.hidden_dim))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.bias is not None:
            self.bias.data.fill_(0)

        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def _prepare_attentional_mechanism_input(self, x):

        Wh1 = torch.matmul(x, self.a[:self.hidden_dim, :])
        Wh2 = torch.matmul(x, self.a[self.hidden_dim:, :])
        e = F.elu(Wh1 + Wh2.T)
        return e

    def forward(self, x, adj):

        h = self.conv(x)

        batch_size, out_channels, height, width = h.shape
        h = h.view(batch_size * height * width, out_channels)

        e = self._prepare_attentional_mechanism_input(h)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj.to_dense() > 0, e, zero_vec)

        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, training=self.training)
        h_pass = torch.matmul(attention, h)

        output_data = h_pass

        output_data = F.elu(output_data)
        output_data = F.normalize(output_data, p=2, dim=1)

        if self.bias is not None:
            output_data = output_data + self.bias

        return output_data

    class KanGNN(torch.nn.Module):
        def __init__(self, in_feat, hidden_feat, out_feat, grid_feat, num_layers, use_bias=False):
            super().__init__()
            self.num_layers = num_layers
            self.lin_in = nn.Linear(in_feat, hidden_feat, bias=use_bias)
            self.lins = torch.nn.ModuleList()
            for i in range(num_layers):
                self.lins.append(FKANLayer(hidden_feat, hidden_feat, grid_feat, addbias=use_bias))
            self.lins.append(nn.Linear(hidden_feat, out_feat, bias=False))

        def forward(self, x, adj):
            x = self.lin_in(x)
            for layer in self.lins[:self.num_layers - 1]:
                x = layer(spmm(adj, x))
            x = self.lins[-1](x)
            return x
