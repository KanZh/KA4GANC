from torch import nn
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import scipy.sparse as sp
from utils import scRNADataset, load_data, adj2saprse_tensor, Evaluation,  Network_Statistic
import pandas as pd
from model import KA4GANC
from PytorchTools import EarlyStopping
import numpy as np
import random
import glob
import os

import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=1e-2, help='Initial learning rate.')
parser.add_argument('--epochs', type=int, default=60,help='Number of epoch.')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--hidden_dim', type=int, default=[128,64,32], help='The dimension of hidden layer')
parser.add_argument('--output_dim', type=int, default=64, help='The dimension of latent layer')
parser.add_argument('--batch_size', type=int, default=512, help='The size of each batch')
parser.add_argument('--loop', type=bool, default=False, help='whether to add self-loop in adjacent matrix')
parser.add_argument('--seed', type=int, default=8, help='Random seed')

args = parser.parse_args()
seed = args.seed
random.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

data_type = 'hESC'
num = 500
net_type = 'Specific'

def embed2file(tf_embed,tg_embed,gene_file,tf_path,target_path):
    tf_embed = tf_embed.cpu().detach().numpy()
    tg_embed = tg_embed.cpu().detach().numpy()

    gene_set = pd.read_csv(gene_file, index_col=0)

    tf_embed = pd.DataFrame(tf_embed,index=gene_set['Gene'].values)
    tg_embed = pd.DataFrame(tg_embed, index=gene_set['Gene'].values)

    tf_embed.to_csv(tf_path)
    tg_embed.to_csv(target_path)


density = Network_Statistic(data_type,num,net_type)
exp_file = './Specific Dataset/'+data_type+'/TFs+'+str(num)+'/BL--ExpressionData.csv'
tf_file = './Specific Dataset/'+data_type+'/TFs+'+str(num)+'/TF.csv'
target_file = './Specific Dataset/'+data_type+'/TFs+'+str(num)+'/Target.csv'

train_file = './Specific Dataset/Train_validation_test/'+data_type+' '+str(num)+'/Train_set.csv'
test_file = './Specific Dataset/Train_validation_test/'+data_type+' '+str(num)+'/Test_set.csv'
val_file = './Specific Dataset/Train_validation_test/'+data_type+' '+str(num)+'/Validation_set.csv'


tf_embed_path = r'Result/'+data_type+' '+str(num)+'/Channel1.csv'
target_embed_path = r'Result/'+data_type+' '+str(num)+'/Channel2.csv'
if not os.path.exists('Result/'+data_type+' '+str(num)):
    os.makedirs('Result/'+data_type+' '+str(num))


data_input = pd.read_csv(exp_file,index_col=0)
loader = load_data(data_input)
feature = loader.exp_data()
tf = pd.read_csv(tf_file,index_col=0)['index'].values.astype(np.int64)
target = pd.read_csv(target_file,index_col=0)['index'].values.astype(np.int64)
feature = torch.from_numpy(feature)
tf = torch.from_numpy(tf)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
data_feature = feature.to(device)
tf = tf.to(device)

train_data = pd.read_csv(train_file, index_col=0).values
validation_data = pd.read_csv(val_file, index_col=0).values
test_data = pd.read_csv(test_file, index_col=0).values

train_load = scRNADataset(train_data, feature.shape[0])
adj = train_load.Adj_Generate(tf,loop=args.loop)


adj = adj2saprse_tensor(adj)


train_data = torch.from_numpy(train_data)
test_data = torch.from_numpy(test_data)
val_data = torch.from_numpy(validation_data)

model = KA4GANC(
    input_dim=feature.size()[1],
    hidden1_dim=args.hidden_dim[0],
    hidden2_dim=args.hidden_dim[1],
    hidden3_dim=args.hidden_dim[2],
    output_dim=args.output_dim,
    num_head1=args.num_head,
    alpha=args.alpha,
    device=device,
)


adj = adj.to(device)
model = model.to(device)
train_data = train_data.to(device)
test_data = test_data.to(device)
validation_data = val_data.to(device)


optimizer = Adam(model.parameters(), lr=args.lr)
scheduler = StepLR(optimizer, step_size=1, gamma=0.99)

model_path = 'model/'
if not os.path.exists(model_path):
    os.makedirs(model_path)

total_start_time = time.time()

for epoch in range(args.epochs):
    epoch_start_time = time.time()
    running_loss = 0.0

    for train_x, train_y in DataLoader(train_load, batch_size=args.batch_size, shuffle=True):
        model.train()
        optimizer.zero_grad()

        train_y = train_y.to(device).view(-1, 1)
        pred = model(data_feature, adj, train_x)
        pred = torch.sigmoid(pred)
        loss_BCE = F.binary_cross_entropy(pred, train_y)

        loss_BCE.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss_BCE.item()

    model.eval()
    score = torch.sigmoid(score)

    AUC, AUPR, AUPR_norm = Evaluation(y_pred=score, y_true=validation_data[:, -1])

    epoch_end_time = time.time() 
    epoch_time = epoch_end_time - epoch_start_time

    print(f'Epoch: {epoch + 1}, train loss: {running_loss}, AUC: {AUC:.3f}, AUPR: {AUPR:.3f}, Epoch Time: {epoch_time:.2f} seconds')

total_end_time = time.time()
total_training_time = total_end_time - total_start_time
print(f'Total training time: {total_training_time:.2f} seconds')

torch.save(model.state_dict(), model_path + data_type+' '+str(num)+'.pkl')
model.load_state_dict(torch.load(model_path + data_type+' '+str(num)+'.pkl'))
model.eval()
tf_embed, target_embed = model.get_embedding()
embed2file(tf_embed, target_embed, target_file, tf_embed_path, target_embed_path)

score = torch.sigmoid(score)

AUC, AUPR, AUPR_norm = Evaluation(y_pred=score, y_true=test_data[:, -1])

print(f'AUC: {AUC}, AUPRC: {AUPR}')