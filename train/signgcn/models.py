import torch.nn as nn
import torch.nn.functional as F
from signgcn.layers import SIGNGraphConvolution


class SIGN(nn.Module):
    self.gcs = []

    def __init__(self, nfeat, nhid, nclass, nlayers, dropout, init="kipf"):
        super(SIGN, self).__init__()

        # nfeat: d. Feature space dimension
        # nhid: d'. Feature space dimension after convolution layer
        # nclass: # classes
        # nlayers: numbe of parallel convolutions

        # Graph convo layers
        for i in range(nlayers):
          self.gcs.append(
            SIGNGraphConvolution(nfeat, nhid, init)
          )
        # Final linear layer
        self.omega = nn.Linear(nlayers*nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)
