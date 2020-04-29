import dgl
import dgl.function as fn
from dgl.nn.pytorch.softmax import edge_softmax
import numpy as np
import torch as th
import torch.nn.functional as F
import torch.nn as nn


class LGCN(nn.Module):
    def __init__(self, d_loc, d_ctx, d_h, d_s, d_m):
        self.W4 = nn.Linear(d_loc, d_h, bias = False)
        self.W5 = nn.Linear(d_ctx, d_h, bias = False)
        self.W6 = nn.Linear(d_loc + d_ctx + d_h, d_h, bias=False)
        self.W7 = nn.Linear(d_loc + d_ctx + d_h, d_h, bias=False)
        self.W8 = nn.Linear(d_s, d_h, bias=False)
        self.W9 = nn.Linear(d_loc + d_ctx + d_h, d_m, bias=False)
        self.W10 = nn.Linear(d_s, d_m, bias=False)
        self.W11 = nn.Linear(d_ctx, d_ctx, bias=False)
        self.W11b = nn.Linear(d_m, d_ctx, bias=False)
    

    def forward(self, g, h, ctx, c):
        fuse = self.W4(h) * self.W5(ctx)
        cat = th.cat([h, ctx, fuse])
        src_ctx = self.W7(cat) * self.W8(c)
        dst_ctx = self.W6(fuse)
        g.srcdata.update({"s_e": src_ctx})
        g.dstdata.update({"d_e": dst_ctx})
        g.apply_edges(fn.u_dot_v("s_e", "d_e", "e"))
        e = g.edata.pop('e')
        g.edata['a'] = edge_softmax(g, e)
        g.ndata['ft'] = self.W9(cat) * self.W10(c)
        g.update_all(fn.u_mul_e('ft', 'a', 'm'), fn.sum('m', 's'))
        g.ndata['ctx'] = self.W11(g.ndata['ctx']) + self.W11b(g.ndata['s'])

        rst = g.ndata['ctx']

        return rst



class TextEmbedder(nn.Module):

    def __init__(self):
        pass

    def forward(self, x):
        pass


