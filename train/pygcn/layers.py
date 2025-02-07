import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True, init="kipf"):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters(init)

    def reset_parameters(self, init):
        
        if init == "kipf":
          stdv = 1. / math.sqrt(self.weight.size(1))
          self.weight.data.uniform_(-stdv, stdv)
          if self.bias is not None:
              self.bias.data.uniform_(-stdv, stdv)
        
        elif init == "xavier":
          torch.nn.init.xavier_uniform_(self.weight)
          if self.bias is not None:
              torch.nn.init.zeros_(self.bias)
        
        elif init == "kaiming":
          torch.nn.init.kaiming_uniform_(self.weight)
          if self.bias is not None:
              torch.nn.init.zeros_(self.bias)
        
        else:
          print("Unrecognized initialization schema!")

    def forward(self, input, adj):
        # Input: features matrix. X, for instance
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
