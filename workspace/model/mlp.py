import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self,
                input_dim,
                hidden_dim,
                output_dim,
                num_layers,
                dropout,
                batchnorm = False):
        super(MLP,self).__init__()

        #self.training = training
        self.linears = nn.ModuleList()
        self.linears.append(nn.Linear(input_dim,hidden_dim))

        self.batchnorm = batchnorm
        if self.batchnorm:
            self.bns = nn.ModuleList()
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        for i in range(num_layers - 2):
            self.linears.append(nn.Linear(hidden_dim,hidden_dim))

            if self.batchnorm:
                self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        self.linears.append(nn.Linear(hidden_dim,output_dim))

        self.dropout = dropout
    
    def reset_parameter(self):
        for linear in self.linears:
            linear.reset_parameter()
        if self.batchnorm:
            for bn in self.bns:
                bn.reset_parameter()
    
    def forward(self,x):
        for idx,linear in enumerate(self.linears[:-1]):
            x = linear(x)
            if self.batchnorm:
                x = self.bns[i](x)
            x = F.leaky_relu(x)
            x = F.dropout(x,p = self.dropout)
        
        x = self.linears[-1](x)
        return F.log_softmax(x,dim=-1)




