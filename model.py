import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optm
from torch.nn import CosineSimilarity
from Code.kan_conv import KANConv2DLayer
from KANGNN import KanGNN
from FKAN import FourierKANLayer
from GATConv import GATKConv



class KA4GANC(nn.Module):
    def __init__(self,input_dim,hidden1_dim,hidden2_dim,hidden3_dim,output_dim,num_head1,alpha,device):
        super(KA4GANC, self).__init__()
        self.input_dim = input_dim
        self.hidden1_dim = num_head1*hidden1_dim
        self.hidden2_dim = hidden2_dim
        self.hidden3_dim = hidden3_dim

        self.num_head1 = num_head1
        self.device = device
        self.alpha = alpha

        self.GKan1 = nn.ModuleList([
            GKAN(self.input_dim, 128, self.hidden2_dim, 150, 1) for _ in range(1)
        ])
        self.add_module('GKAN', self.GKan1)

        self.GKan2 = nn.ModuleList([
            GKAN(self.hidden2_dim, 128, self.hidden3_dim, 150, 1) for _ in range(1)
        ])
        self.add_module('GKAN', self.GKan2)

        self.GKan3 = nn.ModuleList([
            GKAN(self.hidden3_dim, 128, 1, 150, 1) for _ in range(1)
        ])
        self.add_module('GKAN', self.GKan3)

        self.GATKConv = Conv(
            in_channels=1,
            out_channels=64,
            kernel_size=(1),
            hidden_dim = self.hidden2_dim
        )
        self.add_module('ConvAttentionLayer',self.GATKConv)

        self.tf_linear1 = nn.Linear(hidden2_dim,hidden3_dim)
        self.target_linear1 = nn.Linear(hidden2_dim,hidden3_dim)

        self.tf_linear2 = nn.Linear(hidden3_dim,output_dim)
        self.target_linear2 = nn.Linear(hidden3_dim, output_dim)

        self.reset_parameters()

    def reset_parameters(self):
        for attention in self.ConvLayer1:
            attention.reset_parameters()

        nn.init.xavier_uniform_(self.tf_linear1.weight,gain=1.414)
        nn.init.xavier_uniform_(self.target_linear1.weight, gain=1.414)
        nn.init.xavier_uniform_(self.tf_linear2.weight, gain=1.414)
        nn.init.xavier_uniform_(self.target_linear2.weight, gain=1.414)



    def encode(self,x,adj):

        x = self.GKan1(x,adj)
        x = F.leaky_relu(x)
        x = self.GKan2(x,adj)
        x = F.leaky_relu(x)
        x = self.GKan3(x,adj)
        x = F.leaky_relu(x)
        x = x.reshape((1,1,x.shape[0],x.shape[1]))
        out = self.GATKConv(x,adj)

        return out


    def decode(self,tf_embed,target_embed):


        prob = torch.mul(tf_embed, target_embed)
        prob = torch.sum(prob,dim=1).view(-1,1)

        return prob


    def forward(self,x,adj,train_sample):

        embed = self.encode(x,adj)
        tf_embed = self.tf_linear1(embed)
        tf_embed = F.dropout(tf_embed,p=0.01)
        tf_embed = self.tf_linear2(tf_embed)
        tf_embed = F.dropout(tf_embed, p=0.01)
        target_embed = self.target_linear1(embed)
        target_embed = F.dropout(target_embed, p=0.01)
        target_embed = self.target_linear2(target_embed)
        target_embed = F.dropout(target_embed, p=0.01)
        self.tf_ouput = tf_embed
        self.target_output = target_embed

        train_tf = tf_embed[train_sample[:,0]]
        train_target = target_embed[train_sample[:, 1]]

        pred = self.decode(train_tf, train_target)

        return pred

    def get_embedding(self):
        return self.tf_ouput, self.target_output

