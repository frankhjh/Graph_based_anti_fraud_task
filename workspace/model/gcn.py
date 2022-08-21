import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN_Net(nn.Module):

    def __init__(self,input_dim,hidden_dim,num_class,dropout):
        super(GCN_Net,self).__init__()
        self.conv1 = GCNConv(input_dim,hidden_dim)
        self.conv2 = GCNConv(hidden_dim,hidden_dim)
        self.conv3 = GCNConv(hidden_dim,num_class)
        self.dropout = dropout

    def forward(self,graph_data):
        x,edge_index = graph_data['x'],graph_data['edge_index']

        x = self.conv1(x,edge_index)
        x = F.leaky_relu(x)
        x = F.dropout(x,p= self.dropout)
        x = self.conv2(x,edge_index)
        x = F.leaky_relu(x)
        x = F.dropout(x,p = self.dropout)
        x = self.conv3(x,edge_index)

        return F.log_softmax(x,dim=1)




