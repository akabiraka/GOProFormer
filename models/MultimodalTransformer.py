import sys
sys.path.append("../GOProFormer")
import numpy as np
import torch
import torch.nn.functional as F
from transformer.config import Config
from transformer.factory import build_transformer_model

class Model(torch.nn.Module):
    def __init__(self, config:Config) -> None:
        super(Model, self).__init__()
        self.config = config 

        self.GOTopoTransformer = build_transformer_model(config=config, decoder=None) # returns only node embeddings
        self.term_embedding_layer = TermEmbeddingLayer(config)

        self.seq_projection_layer = SeqProjectionLayer(config.esm1b_embed_dim, config.embed_dim, config.dropout)

        self.prediction_refinement_layer = PredictionRefinementLayer(config.vocab_size, config.vocab_size, config.dropout)
        self.adj_prediction_layer = AdjMatPredictionLayer()
        

    def forward(self, terms, rel_mat, seqs):
        """ term_nodes: [n_nodes, n_samples, 768]
            rel_mat: [n_nodes, n_nodes]
            seqs: [batch_size, 768]
        """
        seqs = self.seq_projection_layer(seqs) #[batch_size, embed_dim]
        # print(f"seqs_reps: {seqs.shape}")
        
        terms = self.term_embedding_layer(x=terms) # shape: [n_nodes, embed_dim]
        terms = self.GOTopoTransformer(x=terms, key_padding_mask=None, attn_mask=rel_mat)
        # print(f"terms_reps: {terms.shape}")
        
        scores = self.prediction_refinement_layer(seqs, terms)
        return scores



class PredictionRefinementLayer(torch.nn.Module):
    def __init__(self, inp_embed_dim, out_embed_dim, dropout=0.3) -> None:
        super(PredictionRefinementLayer, self).__init__()
        self.dropout = dropout
        # self.w1 = torch.nn.Linear(inp_embed_dim, out_embed_dim)

    def forward(self, seqs_reps, terms_reps):
        scores = seqs_reps.matmul(terms_reps.t()) # shape: n_seqs, n_terms
        # scores = self.w1(scores)

        return scores

class AdjMatPredictionLayer(torch.nn.Module):
    def __init__(self) -> None:
        super(AdjMatPredictionLayer, self).__init__()

    def forward(self, nodes):
        adj = nodes.matmul(nodes.t())
        print(adj.shape)
        return adj



class TermEmbeddingLayer(torch.nn.Module):
    def __init__(self, config:Config) -> None:
        super(TermEmbeddingLayer, self).__init__()
        self.node_proj_layer = ProjectionLayer(config.esm1b_embed_dim, config.embed_dim, config.dropout)

    def forward(self, x):
        #n_nodes, n_samples, esm1b_embed_dim = x.shape
        
        #nodes = []
        #for j in range(n_nodes):
        #    rep = self.node_proj_layer(x[j]) # n_samples, embed_dim
        #    # print(rep.shape)
        #    nodes.append(rep)
        #
        #nodes = torch.stack(nodes)
        nodes = self.node_proj_layer(x)
        #print(nodes.shape)

        return nodes # shape: [n_nodes, embed_dim]



class ProjectionLayer(torch.nn.Module):
    def __init__(self, inp_embed_dim, out_embed_dim, dropout=0.3):
        super(ProjectionLayer, self).__init__()
        self.dropout = dropout
        self.attn_linear = torch.nn.Linear(inp_embed_dim, 1)
        self.projection = torch.nn.Linear(inp_embed_dim, out_embed_dim)

    def forward(self, last_hidden_state):
        """last_hidden_state (torch.Tensor): shape [batch_size, seq_len, dim_embed]"""
        # x = torch.mean(x, dim=1) #global average pooling. shape [batch_size, dim_embed]
        activation = torch.tanh(last_hidden_state) # [batch_size, seq_len, dim_embed]

        score = self.attn_linear(activation) # [batch_size, seq_len, 1]      
        weights = torch.softmax(score, dim=1) # [batch_size, seq_len, 1]
        seq_reps = torch.sum(weights * last_hidden_state, dim=1)  # [batch_size, dim_embed]
        seq_reps = self.projection(seq_reps)
        # seq_reps = F.relu(seq_reps)
        # seq_reps = F.dropout(seq_reps, p=self.dropout)
        return seq_reps



class SeqProjectionLayer(torch.nn.Module):
    def __init__(self, inp_embed_dim, out_embed_dim, dropout_p=0.3) -> None:
        super(SeqProjectionLayer, self).__init__()
        self.dropout_p = dropout_p
        self.projection = torch.nn.Linear(inp_embed_dim, out_embed_dim)

    def forward(self, x):
        return F.dropout(F.relu(self.projection(x)), self.dropout_p)


def count_parameters(model):
    n_trainable_weights = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return n_trainable_weights


def train(model, data_loader, terms_graph, label_pred_criterion, optimizer, device):
    model.train()
    train_loss = 0.0

    for i, (uniprot_ids, seqs, y_true) in enumerate(data_loader):
        seqs, y_true = seqs.to(device), y_true.to(device)
        # print(y_true.shape, seqs.shape, uniprot_ids)

        graph = terms_graph.get(uniprot_ids)
        terms, adj_matrix = graph["nodes"].to(device), graph["adjacency_rel_matrix"].to(device)

        model.zero_grad(set_to_none=True)
        y_pred = model(terms, adj_matrix, seqs)
        
        # batch_loss, _ = compute_loss(y_pred, y_true, criterion) 
        loss = label_pred_criterion(y_pred, y_true)

        loss.backward()
        optimizer.step()
        
        train_loss = train_loss + loss.item()
        # print(f"    train batch: {i}, loss: {loss.item()}")
        # if i==5: break
    return train_loss/len(data_loader)



@torch.no_grad()
def val(model, data_loader, terms_graph, label_pred_criterion, device):
    model.eval()
    val_loss = 0.0
    pred_scores, true_scores = [], []

    for i, (uniprot_ids, seqs, y_true) in enumerate(data_loader):
        seqs, y_true = seqs.to(device), y_true.to(device)
        # print(y_true.shape)

        graph = terms_graph.get(uniprot_ids)
        terms, adj_matrix = graph["nodes"].to(device), graph["adjacency_rel_matrix"].to(device)

        model.zero_grad(set_to_none=True)
        y_pred = model(terms, adj_matrix, seqs)
        
        # batch_loss, _ = compute_loss(y_pred, y_true, criterion) 
        loss = label_pred_criterion(y_pred, y_true)

        val_loss = val_loss + loss.item()
        
        pred_scores.append(torch.sigmoid(y_pred).detach().cpu().numpy())
        true_scores.append(y_true.detach().cpu().numpy())

        # print(f"    val batch: {i}, loss: {loss.item()}")
        # if i==5: break
    true_scores, pred_scores = np.vstack(true_scores), np.vstack(pred_scores)
    # print(true_scores.shape, pred_scores.shape)
    return val_loss/len(data_loader), true_scores, pred_scores



