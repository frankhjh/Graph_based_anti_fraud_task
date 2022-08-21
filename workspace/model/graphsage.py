import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class GraphSAGE_Net(nn.Module):

    def __init__(self,input_dim,hidden_dim,num_class,dropout):
        super(GraphSAGE_Net,self).__init__()
        self.sage1 = SAGEConv(input_dim,hidden_dim)
        self.sage2 = SAGEConv(hidden_dim,num_class)
        self.dropout = dropout

    def forward(self,graph_data):
        x,edge_index = graph_data['x'],graph_data['edge_index']

        x = self.sage1(x,edge_index)
        x = F.leaky_relu(x)
        x = F.dropout(x,p = self.dropout)
        x = self.sage2(x,edge_index)

        return F.log_softmax(x,dim=1)