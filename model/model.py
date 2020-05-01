from typing import List
from typing import Dict
from typing import Tuple
import os
import shutil
import logging


import dgl
import dgl.function as fn
from dgl.nn.pytorch.softmax import edge_softmax

import numpy as np
import networkx as nx

import torch as th
import torch.nn.functional as F
import torch.nn as nn
from .gnn import LGCNLayer
from .config import cfg
from .encoder import Encoder
from .decoder import Decoder

logger = logging.getLogger(__name__)

device = th.device("cuda" if th.cuda.is_available() else "cpu")


class GSCAN_model(nn.Module):
    def __init__(self, pad_idx, target_eos_idx, input_vocab_size, target_vocab_size):
        super().__init__()

        
        # \TODO if we want to add embedding here.
        # if cfg.INIT_WRD_EMB_FROM_FILE:
        #     embeddingsInit = np.load(cfg.WRD_EMB_INIT_FILE)
        #     assert embeddingsInit.shape == (num_vocab-1, cfg.WRD_EMB_DIM)
        # else:
        #     embeddingsInit = np.random.uniform(
        #         low=-1, high=1, size=(num_vocab-1, cfg.WRD_EMB_DIM))

        self.num_vocab = input_vocab_size



        self.encoder = Encoder(pad_idx, input_vocab_size)
        self.lgcn = LGCNLayer()
        self.decoder = Decoder(target_vocab_size, pad_idx)


        self.loss_criterion = nn.NLLLoss(ignore_index = pad_idx)
        self.tanh = nn.Tanh()
        self.target_eos_idx = target_eos_idx
        self.target_pad_idx = pad_idx
        self.auxiliary_task = cfg.AUXILIARY_TASK


        self.output_directory = cfg.OUTPUT_DIRECTORY
        self.trained_iterations = 0
        self.best_iteration = 0
        self.best_exact_match = 0
        self.best_accuracy = 0

        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")
    
    def nonzero_extractor(self, x):
        lx = []
        for i in range(x.size(0)):
            sum_x = th.sum(x[i], dim=-1)
            lx.append(x[i, sum_x.gt(0), :])
        return lx

    
    @staticmethod
    def remove_start_of_sequence(input_tensor):
        """Get rid of SOS-tokens in targets batch and append a padding token to each example in the batch."""
        batch_size, max_time = input_tensor.size()
        input_tensor = input_tensor[:, 1:]
        output_tensor = th.cat([input_tensor, th.zeros(batch_size, device=device, dtype=th.long).unsqueeze(
            dim=1)], dim=1)
        return output_tensor

    
    def get_metrics(self, target_scores, targets):
        """
        :param target_scores: probabilities over target vocabulary outputted by the model, of size
                              [batch_size, max_target_length, target_vocab_size]
        :param targets:  ground-truth targets of size [batch_size, max_target_length]
        :return: scalar float of accuracy averaged over sequence length and batch size.
        """
        with th.no_grad():
            targets = self.remove_start_of_sequence(targets)
            mask = (targets != self.target_pad_idx).long()
            total = mask.sum().data.item()
            predicted_targets = target_scores.max(dim=2)[1]
            equal_targets = th.eq(targets.data, predicted_targets.data).long()
            match_targets = (equal_targets * mask)
            match_sum_per_example = match_targets.sum(dim=1)
            expected_sum_per_example = mask.sum(dim=1)
            batch_size = expected_sum_per_example.size(0)
            exact_match = 100. * (match_sum_per_example == expected_sum_per_example).sum().data.item() / batch_size
            match_targets_sum = match_targets.sum().data.item()
            accuracy = 100. * match_targets_sum / total
        return accuracy, exact_match

    
    

    
    def forward(self, cmd_batch, situation_batch, tgt_batch):
        '''
        cmd_batch[0]: batchsize x max_length
        cmd_batch[1]: batchsize
        situation_batch[0]: batchsize x grid x grid x k
        
        '''
        batchSize = cmd_batch[0].size(0)
        print("batch size is ", batchSize)
        cmdIndices, cmdLengths = cmd_batch[0], cmd_batch[1]
        tgtIndices, tgtLengths = tgt_batch[0], tgt_batch[1]

        # LSTM
        cmd_out, cmd_h = self.encoder(cmdIndices, cmdLengths)
        print("cmd_out size is ", cmd_out.size())
        print("cmd_h size is ", cmd_h.size())

        # Building computation graph for LGCN
        xs = self.nonzero_extractor(situation_batch)
        gs = []
        situations_lengths = []
        graph_membership = []
        dgl_gs =[dgl.DGLGraph() for _ in range(batchSize)]
        for i, x in enumerate(xs):
            gs.append(nx.complete_graph(x.size(0)).to_directed())
            situations_lengths.append(x.size(0))
            graph_membership += [i for _ in range(x.size(0))]
        for i in range(len(dgl_gs)):
            dgl_gs[i].from_networkx(gs[i])
        batch_g = dgl.batch(dgl_gs)
        situation_X = th.cat(xs, dim=0)
        graph_membership = th.tensor(graph_membership, dtype=th.long, device=self.device)


        #LGCN
        situation_out = self.lgcn(situation_X, batch_g, cmd_h, cmd_out, cmdLengths, batchSize, graph_membership)
        situation_out = th.nn.utils.rnn.pad_sequence(situation_out, batch_first=True)


        #Decoder
        decoder_output, context_situation = self.decoder(tgtIndices, tgtLengths, initial_hidden = cmd_h, encoded_commands=cmd_out, commands_lengths=cmdLengths,
        encoded_situations=situation_out, situations_lengths=situations_lengths)


        if self.auxiliary_task:
            target_position_scores = self.auxilaiary_task_forward(context_situation)
        else:
            target_position_scores = th.zeros(1), th.zeros(1)
        
        return (decoder_output.transpose(0, 1), target_position_scores) #decoder_output shape: [batch_size, max_target_seq_length, target_vocab_size]
    

    def get_loss(self, target_score, target):
        target = self.remove_start_of_sequence(target) #\TODO make sure include sos in target

        # Calculate the loss
        _, _, vocabulary_size = target_score.size()
        target_score_2d = target_score.reshape(-1, vocabulary_size)
        loss = self.loss_criterion(target_score_2d, target.view(-1))
        return loss

    
    def update_state(self, is_best: bool, accuracy=None, exact_match=None) -> {}:
        self.trained_iterations += 1
        if is_best:
            self.best_exact_match = exact_match
            self.best_accuracy = accuracy
            self.best_iteration = self.trained_iterations
    
    def save_checkpoint(self, file_name: str, is_best: bool, optimizer_state_dict: dict) -> str:
        """
        :param file_name: filename to save checkpoint in.
        :param is_best: boolean describing whether or not the current state is the best the model has ever been.
        :param optimizer_state_dict: state of the optimizer.
        :return: str to path where the model is saved.
        """
        path = os.path.join(self.output_directory, file_name)
        state = self.get_current_state()
        state["optimizer_state_dict"] = optimizer_state_dict
        th.save(state, path)
        if is_best:
            best_path = os.path.join(self.output_directory, 'model_best.pth.tar')
            shutil.copyfile(path, best_path)
        return path 




