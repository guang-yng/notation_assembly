import torch
import torch.nn as nn
from utils.constants import node_classes_dict


class MLP(torch.nn.Module):

    def __init__(self, config):
        super(MLP, self).__init__()
        self.config = config
        self.class_embed = nn.Embedding(max(node_classes_dict.values())+1, config.MODEL.EMBEDDING_DIM)
        self.MLP = nn.Sequential()
        for idx, dim in enumerate(self.config.MODEL.MLP_CONFIG):
            if idx == 0:
                self.MLP.append(nn.Linear(config.MODEL.EMBEDDING_DIM*2+8, dim))
                self.MLP.append(nn.ReLU())
            else:
                self.MLP.append(nn.Linear(config.MODEL.MLP_CONFIG[idx-1], dim))
                self.MLP.append(nn.ReLU())
        self.head = nn.Linear(config.MODEL.MLP_CONFIG[-1], 1)

    def forward(self, batch, apply_sigmoid=False):
        source_cls_embed = self.class_embed(batch['source_class'])
        target_cls_embed = self.class_embed(batch['target_class'])
        source_embed = torch.cat([batch['source_bbox'], source_cls_embed], dim=-1)
        target_embed = torch.cat([batch['target_bbox'], target_cls_embed], dim=-1)

        combined_embed = torch.cat([source_embed, target_embed], dim=-1)
        combined_embed = self.MLP(combined_embed)
        output = self.head(combined_embed)
        if apply_sigmoid:
            return torch.sigmoid(output)
        else:
            return output