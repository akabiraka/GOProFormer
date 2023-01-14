import sys
sys.path.append("../GOProFormer")
import torch
import torch.nn as nn

class Decoder(nn.Module):
    pass

class PairwiseDistanceDecoder(Decoder):
    def __init__(self):
        super(PairwiseDistanceDecoder, self).__init__()

    def forward(self, h, sigmoid=True):
        l2_dist = torch.cdist(h, h, p=1)
        return torch.sigmoid(l2_dist) if sigmoid else l2_dist


class GraphClassifier(Decoder):
    """Same as sequence classifier"""
    def __init__(self, dim_embed, n_classes, dropout=0.3):
        super(GraphClassifier, self).__init__()
        self.attn_linear = torch.nn.Linear(dim_embed, 1)
        self.classifier = nn.Linear(dim_embed, n_classes)

    def forward(self, last_hidden_state):
        """last_hidden_state (torch.Tensor): shape [batch_size, seq_len, dim_embed]"""
        activation = torch.tanh(last_hidden_state) # [batch_size, seq_len, dim_embed]

        score = self.attn_linear(activation) # [batch_size, seq_len, 1]      
        weights = torch.softmax(score, dim=1) # [batch_size, seq_len, 1]
        last_layer_learned_rep = torch.sum(weights * last_hidden_state, dim=1)  # [batch_size, dim_embed]

        y = self.classifier(last_layer_learned_rep) # [batch_size, n_classes]
        return y, last_layer_learned_rep


class NodeClassifier(Decoder):
    def __init__(self, dim_embed, n_classes, dropout=0.3) -> None:
        super(NodeClassifier, self).__init__()
        self.classifier = nn.Linear(dim_embed, n_classes)

    def forward(self, last_hidden_state):
        """last_hidden_state: [n_nodes, dim_embed]"""
        y = self.classifier(last_hidden_state)
        return y, None
