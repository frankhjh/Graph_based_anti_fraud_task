import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GAT_Net(nn.Module):
    def __init__(self,input_dim,hidden_dim,num_class,num_head,dropout):
        super(GAT_Net,self).__init__()
        self.gat1 = GATConv(input_dim,hidden_dim,heads=num_head)
        self.gat2 = GATConv(hidden_dim * num_head,num_class)
        self.dropout = dropout
    
    def forward(self,graph_data):
        x,edge_index = graph_data['x'],graph_data['edge_index']
        x = self.gat1(x,edge_index)
        x = F.leaky_relu(x)
        x = F.dropout(x,p=self.dropout)
        x = self.gat2(x,edge_index)

        return F.log_softmax(x,dim=1)
        