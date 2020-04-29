import dgl
import dgl.function as fn
from dgl.nn.pytorch.softmax import edge_softmax
import numpy as np
import torch as th
import torch.nn.functional as F
import torch.nn as nn
from .nn import *


class GSCAN_model(nn.Module):
    def __init__(self, T, d_loc, d_ctx, d_h):
        self.LGCN = LGCN()
        self.T = T
        self.encoder = lambda x: x
        self.decoder = lambda x: x

        self.W12 = nn.Linear(d_loc + d_ctx, d_h, bias=False)
    
    def init_ctx(self):
        return None

    def forward(self, g, h, s):
        c = self.sentence_embed(s)
        
        ctx = self.init_ctx()
        for _ in range(self.T):
            ctx = self.LGCN(g, h, ctx, c)
        
        ctx_ft = self.W12(th.cat(h, ctx))

        res = self.decoder.decode(ctx_ft, s)

        return res



