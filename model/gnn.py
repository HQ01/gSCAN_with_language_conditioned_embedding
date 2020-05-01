import dgl
import dgl.function as fn
from dgl.nn.pytorch.softmax import edge_softmax
import numpy as np
import torch as th
import torch.nn.functional as F
import torch.nn as nn

from .config import cfg
from ..utils import *


class LGCNLayer(nn.Module):
    def __init__(self):
        d_x = cfg.SITU_D_FEAT #\TODO infer this value from dataset statistics method
        d_cmd = cfg.SITU_D_CMD
        d_loc = cfg.SITU_D_CTX

        d_ctx = cfg.SITU_D_CTX
        d_h = cfg.SITU_D_CTX
        d_s = cfg.SITU_D_CMD
        d_m = cfg.SITU_D_CTX


        #Iter #
        self.T = cfg.MSG_ITER_NUM

        
        # X_loc init Config
        self.map_x_loc = nn.Linear(d_x, d_loc, bias=False)
        self.x_loc_drop = nn.Dropout(1 - cfg.locDropout)

        ## Command Extraction Config
        self.W1 = nn.Linear(d_cmd, 1, bias=False)
        self.W2_layers = nn.ModuleList()
        for _ in range(self.T):
            self.W2_layers.append(nn.Linear(d_cmd, d_cmd, bias=False))
        self.W3 = nn.Linear(d_cmd, d_cmd)
        
        
        #GNN layer config
        self.W4 = nn.Linear(d_loc, d_h, bias = False)
        self.W5 = nn.Linear(d_ctx, d_h, bias = False)
        self.W6 = nn.Linear(d_loc + d_ctx + d_h, d_h, bias=False)
        self.W7 = nn.Linear(d_loc + d_ctx + d_h, d_h, bias=False)
        self.W8 = nn.Linear(d_s, d_h, bias=False)
        self.W9 = nn.Linear(d_loc + d_ctx + d_h, d_m, bias=False)
        self.W10 = nn.Linear(d_s, d_m, bias=False)
        self.W11 = nn.Linear(d_ctx, d_ctx, bias=False)
        self.W11b = nn.Linear(d_m, d_ctx, bias=False)
        self.W12 = nn.Linear(d_loc + d_ctx, d_h, bias=False)

        
        self.initMem = nn.Parameter(th.randn(1, 1, d_ctx))
    
    def loc_ctx_init(self, xs):
        x_loc = F.noramlize(xs, dim=-1) # could add linear transformation
        x_loc = self.x_loc_drop(self.map_x_loc(x_loc))
        x_loc = F.noramlize(x_loc, dim=-1)
        x_ctx = self.initMem.expand(x_loc.size())

        return x_loc, x_ctx
    
    def extract_textual_command(self, cmd_h, cmd_out, cmdLength, t):
        raw_att =  self.W1(cmd_out * self.W2_layers[t](F.relu(self.W3(cmd_h)))).squeeze(-1) #\TODO cmd_out * sth might be wrong
        mask = sequence_mask(cmdLength)
        #maxlen = raw_att.size(1)
        #mask = th.arange(maxlen)[None, :] < cmdLength[:, None]
        att = masked_softmax(raw_att, mask)
        cmd = th.bmm(att[:, None, :], cmd_out).squeeze(1)

        return cmd





    
    def graph_nn(self, g, h, ctx, c):
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

    

    def forward(self, situation_x, batch_g, cmd_h, cmd_out, cmdLength, batch_size):
        x_loc, x_ctx = self.loc_ctx_init(situation_x)
        for t in range(self.T):
            cmd = self.extract_textual_command(cmd_h, cmd_out, cmdLength, t) #\TODO check whether mixing cmd_h, cmd_out
            x_ctx = self.graph_nn(batch_g, x_loc, x_ctx, cmd)
        
        x_out = self.W12(th.cat([x_loc, x_ctx], dim=-1))
        return x_out


