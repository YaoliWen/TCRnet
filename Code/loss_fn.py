# Lib
import torch
from torch import nn

# varine loss
class VarLoss(nn.Module):
    def __init__(self):
        super(VarLoss, self).__init__()

    def forward(self, attention_all):
        losses = []
        for attention in attention_all:
            batch_size = attention.shape[0] # B
            num_head = attention.shape[1] # n_h
            var = torch.var(attention, dim=-1)
            var_loss = var.sum() / (batch_size * num_head)
            losses.append(var_loss.unsqueeze(dim=0))
        losses = torch.cat(losses)
        return losses