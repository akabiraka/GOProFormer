import sys
sys.path.append("../GOProFormer")
import copy
import torch.nn as nn

from transformer.components import FeedForward, MultiheadAttentionWrapper, Embeddings, PositionalEncoding
from transformer.encoder import EncoderLayer, Encoder
from transformer.decoder import Decoder
from transformer.encoder_decoder import EncoderDecoder
from transformer.config import Config



def build_transformer_model(config:Config, decoder:Decoder):                
    cp = copy.deepcopy
    
    # encoder
    attn = MultiheadAttentionWrapper(config.embed_dim, config.n_attn_heads)
    ff = FeedForward(config.embed_dim, config.dim_ff, config.dropout)
    enc_layer = EncoderLayer(config.embed_dim, cp(attn), cp(ff), config.dropout)
    encoder = Encoder(enc_layer, config.n_encoder_layers)

    # example decoders
    # decoder = TanhClassifier(config.embed_dim, config.n_classes, config.dropout)
    # decoder = NodeClassifier(config.embed_dim, config.n_classes, config.dropout)

    # full model
    node_embed_layer = Embeddings(config.vocab_size, config.embed_dim) if config.add_node_embed_layer else None
    positional_encoding = PositionalEncoding(config.embed_dim, config.dropout, config.max_num_of_nodes) if config.add_positional_encoding_layer else None
    model = EncoderDecoder(encoder, decoder, node_embed_layer, positional_encoding)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model

# print(build_model(config()))
