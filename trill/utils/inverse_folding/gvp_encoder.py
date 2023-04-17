# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from .features import GVPGraphEmbedding
from .gvp_modules import GVPConvLayer, LayerNorm 
from .gvp_utils import unflatten_graph



class GVPEncoder(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.embed_graph = GVPGraphEmbedding(args)

        node_hidden_dim = (args.node_hidden_dim_scalar,
                args.node_hidden_dim_vector)
        edge_hidden_dim = (args.edge_hidden_dim_scalar,
                args.edge_hidden_dim_vector)
        
        conv_activations = (F.relu, torch.sigmoid)
        self.encoder_layers = nn.ModuleList(
                GVPConvLayer(
                    node_hidden_dim,
                    edge_hidden_dim,
                    drop_rate=args.dropout,
                    vector_gate=True,
                    attention_heads=0,
                    n_message=3,
                    conv_activations=conv_activations,
                    n_edge_gvps=0,
                    eps=1e-4,
                    layernorm=True,
                ) 
            for i in range(args.num_encoder_layers)
        )

    def forward(self, coords, coord_mask, padding_mask, confidence):
        node_embeddings, edge_embeddings, edge_index = self.embed_graph(
                coords, coord_mask, padding_mask, confidence)
        
        for i, layer in enumerate(self.encoder_layers):
            node_embeddings, edge_embeddings = layer(node_embeddings,
                    edge_index, edge_embeddings)

        node_embeddings = unflatten_graph(node_embeddings, coords.shape[0])
        return node_embeddings

class lightning_GVPEncoder(pl.LightningModule):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.embed_graph = GVPGraphEmbedding(args)

        node_hidden_dim = (args.node_hidden_dim_scalar,
                args.node_hidden_dim_vector)
        edge_hidden_dim = (args.edge_hidden_dim_scalar,
                args.edge_hidden_dim_vector)
        
        conv_activations = (F.relu, torch.sigmoid)
        self.encoder_layers = nn.ModuleList(
                GVPConvLayer(
                    node_hidden_dim,
                    edge_hidden_dim,
                    drop_rate=args.dropout,
                    vector_gate=True,
                    attention_heads=0,
                    n_message=3,
                    conv_activations=conv_activations,
                    n_edge_gvps=0,
                    eps=1e-4,
                    layernorm=True,
                ) 
            for i in range(args.num_encoder_layers)
        )

    def forward(self, coords, coord_mask, padding_mask, confidence):
        node_embeddings, edge_embeddings, edge_index = self.embed_graph(
                coords, coord_mask, padding_mask, confidence)
        
        for i, layer in enumerate(self.encoder_layers):
            node_embeddings, edge_embeddings = layer(node_embeddings,
                    edge_index, edge_embeddings)

        node_embeddings = unflatten_graph(node_embeddings, coords.shape[0])
        return node_embeddings

    def training_step(self, batch, batch_idx):
        # torch.cuda.empty_cache()
        # labels, seqs, toks = batch
        # del labels, seqs, batch_idx
        # masked_toks = maskInputs(toks, self.esm)
        # output = self.esm(masked_toks, repr_layers = [-1], return_contacts=False)
        # loss = F.cross_entropy(output['logits'].permute(0,2,1), toks)
        # self.log("loss", loss)
        # del masked_toks, toks
        return {"loss": loss}
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.embed_graph, lr=1e-5)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]