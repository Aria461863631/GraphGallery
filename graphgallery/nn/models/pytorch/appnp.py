import torch.nn as nn
from torch import optim

from graphgallery.nn.models import TorchKeras
from graphgallery.nn.layers.pytorch import APPNPropagation, PPNPropagation, activations
from graphgallery.nn.metrics.pytorch import Accuracy


class APPNP(TorchKeras):
    def __init__(self,
                 in_channels,
                 out_channels,
                 *,
                 alpha=0.1,
                 K=10,
                 ppr_dropout=0.,
                 hids=[64],
                 acts=['relu'],
                 dropout=0.5,
                 weight_decay=5e-4,
                 lr=0.01,
                 bias=True,
                 approximated=True):

        super().__init__()
        lin = []
        lin.append(nn.Dropout(dropout))
        for hid, act in zip(hids, acts):
            lin.append(nn.Linear(in_channels,
                                 hid,
                                 bias=bias))
            lin.append(activations.get(act))
            lin.append(nn.Dropout(dropout))
            in_channels = hid
        lin.append(nn.Linear(in_channels, out_channels, bias=bias))
        lin = nn.Sequential(*lin)
        self.lin = lin
        if approximated:
            self.propagation = APPNPropagation(alpha=alpha, K=K,
                                               dropout=ppr_dropout)
        else:
            self.propagation = PPNPropagation(dropout=ppr_dropout)
        self.compile(loss=nn.CrossEntropyLoss(),
                     optimizer=optim.Adam([dict(params=lin[1].parameters(),
                                                weight_decay=weight_decay),
                                           dict(params=lin[2:].parameters(),
                                                weight_decay=0.)], lr=lr),
                     metrics=[Accuracy()])
        self.act_fn = nn.ReLU()

    def forward(self, x, adj):
        x = self.lin(x)
        x = self.propagation(x, adj)
        return x
