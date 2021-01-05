import torch.nn as nn
import torch
import torch.nn.functional as F
from signgcn.layers import SIGNGraphConvolution

def init_linear(m):
  m.weight.data.normal_(0, 0.01)
  m.bias.data.normal_(0, 0.01)

class SIGN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, nlayers, dropout, init="kipf"):
        super(SIGN, self).__init__()

        # nfeat: d. Feature space dimension
        # nhid: d'. Feature space dimension after convolution layer
        # nclass: # classes
        # nlayers: numbe of parallel convolutions

        # Graph convo layers
        self.gcs = nn.ModuleList()
        for i in range(nlayers):
          self.gcs.append(
            SIGNGraphConvolution(nfeat, nhid, init)
          )
        # Final linear layer
        self.omega = nn.Linear(nlayers*nhid, nclass)
        self.omega.apply(init_linear)
        self.dropout = dropout

    def forward(self, precomp_ax_list):
        if(len(precomp_ax_list) != len(self.gcs)):
          print("There is an inconsistency in the number of layers and number of precomputed adj matrices products with features")
          return None

        prev_x = None
        for i, ax in enumerate(precomp_ax_list):
          x = self.gcs[i](ax)
          #x = F.dropout(x, self.dropout, training=self.training)
          # concat
          if prev_x is not None:
            x = torch.cat([prev_x, x], dim=1)
          prev_x = x

        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.omega(x)
        return F.log_softmax(x, dim=1)
