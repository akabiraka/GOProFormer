import sys
sys.path.append("../GOProFormer")
import torch.nn as nn

class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder=None, node_embed_layer=None, positional_encoding=None):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.node_embed_layer = node_embed_layer
        self.positional_encoding = positional_encoding
    
    def forward(self, x, key_padding_mask=None, attn_mask=None):
        """seq_reps:[batch_size, embed_dim]"""
        if self.node_embed_layer is not None: x = self.node_embed_layer(x)
        if self.positional_encoding is not None: x = self.positional_encoding(x)
        x, _ = self.encoder(x, key_padding_mask, attn_mask)
        if self.decoder is not None: x, _ = self.decoder(x)
        return x
    
    def get_node_embeddings(self, x, with_positional_encoding=False):
        if with_positional_encoding: return self.positional_encoding(self.node_embed_layer(x))
        return self.node_embed_layer(x)
    
    def get_all_layers_attn_weights(self, x, key_padding_mask=None, attn_mask=None):
        if self.node_embed_layer is not None: x = self.node_embed_layer(x)
        if self.positional_encoding is not None: x = self.positional_encoding(x)
        _, all_layers_attn_weights = self.encoder(x, key_padding_mask, attn_mask, return_attn_weights=True)
        return all_layers_attn_weights

    def get_last_layer_learned_rep(self, x, key_padding_mask=None, attn_mask=None):
        if self.decoder is None:
            print("Decoder is set to be None, but it cannot.")
            return
        if self.node_embed_layer is not None: x = self.node_embed_layer(x)
        if self.positional_encoding is not None: x = self.positional_encoding(x)
        x, _ = self.encoder(x, key_padding_mask, attn_mask)
        _, last_layer_learned_rep = self.decoder(x)
        return last_layer_learned_rep