# Directly from m-ionic

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from Bio import SeqIO
from tqdm import tqdm
import urllib.request
from loguru import logger

def download_mionic_checkpoint(cache_dir):
    url = "https://raw.githubusercontent.com/TeamSundar/m-ionic/main/checkpoints/esm2_t33_650M_UR50D_setB_fold1.pt"
    output_dir = os.path.join(cache_dir, "MIonic")
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "esm2_t33_650M_UR50D_setB_fold1.pt")
    
    if os.path.exists(output_path):
        logger.info(f"File already exists: {output_path}. Skipping download.")
        return output_path

    logger.info(f"Downloading checkpoint to: {output_path}")
    urllib.request.urlretrieve(url, output_path)
    return output_path

def mionic(args, per_AA, model, mionic_path):
    device = "cuda" if int(args.GPUs) > 0 else "cpu"
    fasta_file = args.query

    model.load_state_dict(torch.load(mionic_path, map_location=device))
    model.to(device).eval()
    
    # Prepare sigmoid function and label fields
    m = nn.Sigmoid()
    fields = ['CA', 'CO', 'CU', 'FE2', 'FE', 'MG', 'MN', 'PO4', 'SO4', 'ZN', 'null']
    output_dir = os.path.join(args.outdir, f'{args.name}_M-Ionic_output')
    # Create output directory if not exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Initialize data structure to track counts per sequence
    total_counts_per_sequence = {}
    residue_level_predictions = {}
    total_sequences = sum(1 for _ in SeqIO.parse(fasta_file, "fasta"))

    # Process FASTA sequences
    for seq_record in tqdm(SeqIO.parse(fasta_file, "fasta"), total=total_sequences, desc="Performing inference with M-Ionic"):
        match, label = next((item[0][0], item[0][1]) for item in per_AA if item[0][1] == seq_record.id)                                  
        sequence_representations = torch.from_numpy(match).to(dtype=torch.float32, device=device)  # [L, 1280]
        per_residue_preds = []
        # Initialize a dictionary to track counts for this sequence
        sequence_counts = {field: 0 for field in fields if field != 'null'}
        
        # Make predictions for each residue in the sequence
        for residue_repr in sequence_representations:
            emb1 = residue_repr.unsqueeze(0)  # Add batch dimension
            
            with torch.no_grad():
                outputs = model(emb1, mask=None)
                predictions_sigmoid = m(torch.stack(outputs))
                predictions_sigmoid = predictions_sigmoid.unsqueeze(0)
                
                # Check if predictions match the number of fields
                if predictions_sigmoid.size(1) != len(fields):
                    logger.warn(f"Warning: Prediction size mismatch for {seq_record.id}. Skipping this residue.")
                    continue
                
                predictions_binary = torch.round(predictions_sigmoid)
                residue_prediction = {}

                # Aggregate count of 1s per column (ion) for the current sequence
                for ind, ion in enumerate(fields):
                    if ion != 'null':
                        count = predictions_binary[0, ind].cpu().item()  # Indexing single row
                        sequence_counts[ion] += count
                        residue_prediction[ion] = count
                per_residue_preds.append(residue_prediction)
        # Store the counts for this sequence
        total_counts_per_sequence[seq_record.id] = sequence_counts
        residue_level_predictions[seq_record.id] = per_residue_preds

    return total_counts_per_sequence, residue_level_predictions

class Self_Attention(nn.Module):
    def __init__(self, num_hidden, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.attention_head_size = int(num_hidden / num_heads)
        self.all_head_size = self.num_heads * self.attention_head_size

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        #print(x.shape)
        return x.permute(0, 2, 1)

    def forward(self, q, k, v, mask=None):
        q = self.transpose_for_scores(q) # [bsz, heads, protein_len, hid]
        k = self.transpose_for_scores(k)
        v = self.transpose_for_scores(v)

        attention_scores = torch.matmul(q, k.transpose(-1, -2))

        if mask is not None:
            attention_mask = (1.0 - mask) * -10000
            attention_scores = attention_scores + attention_mask.unsqueeze(1).unsqueeze(1)

        attention_scores = nn.Softmax(dim=-1)(attention_scores)

        outputs = torch.matmul(attention_scores, v)

        outputs = outputs.permute(0, 2, 1).contiguous()
        new_output_shape = outputs.size()[:-2] + (self.all_head_size,)
        outputs = outputs.view(*new_output_shape)
        return outputs


class PositionWiseFeedForward(nn.Module):
    def __init__(self, num_hidden, num_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.W_in = nn.Linear(num_hidden, num_ff, bias=True)
        self.W_out = nn.Linear(num_ff, num_hidden, bias=True)

    def forward(self, h_V):
        h = F.leaky_relu(self.W_in(h_V))
        h = self.W_out(h)
        return h

class TransformerLayer(nn.Module):
    def __init__(self, num_hidden = 64, num_heads = 4, dropout = 0.2):
        super(TransformerLayer, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.ModuleList([nn.LayerNorm(num_hidden, eps=1e-6) for _ in range(2)])

        self.attention = Self_Attention(num_hidden, num_heads)
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, mask=None):
        # Self-attention
        dh = self.attention(h_V, h_V, h_V, mask)
        h_V = self.norm[0](h_V + self.dropout(dh))

        # Position-wise feedforward
        dh = self.dense(h_V)
        h_V = self.norm[1](h_V + self.dropout(dh))

        if mask is not None:
            mask = mask.unsqueeze(-1)
            h_V = mask * h_V
        return h_V


class IonicProtein(nn.Module):
    def __init__(self, feature_dim, hidden_dim=128, num_encoder_layers=4, num_heads=4, augment_eps=0.05, dropout=0.2):
        super(IonicProtein, self).__init__()

        # Hyperparameters
        self.augment_eps = augment_eps

        # Embedding layers
        self.input_block = nn.Sequential(
                                         nn.LayerNorm(feature_dim, eps=1e-6)
                                        ,nn.Linear(feature_dim, hidden_dim)
                                        ,nn.LeakyReLU()
                                        )

        self.hidden_block = nn.Sequential(
                                          nn.LayerNorm(hidden_dim, eps=1e-6)
                                         ,nn.Dropout(dropout)
                                         ,nn.Linear(hidden_dim, hidden_dim)
                                         ,nn.LeakyReLU()
                                         ,nn.LayerNorm(hidden_dim, eps=1e-6)
                                         )

        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_encoder_layers)
        ])

        # ion-specific layers
        # dict_keys(['CA', 'CO', 'CU', 'FE2', 'FE', 'MG', 'MN', 'PO4', 'SO4', 'ZN', 'null'])
        self.FC_CA_1 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.FC_CA_2 = nn.Linear(hidden_dim, 1, bias=True)
        self.FC_CO_1 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.FC_CO_2 = nn.Linear(hidden_dim, 1, bias=True)
        self.FC_CU_1 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.FC_CU_2 = nn.Linear(hidden_dim, 1, bias=True)
        self.FC_FE2_1 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.FC_FE2_2 = nn.Linear(hidden_dim, 1, bias=True)
        self.FC_FE_1 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.FC_FE_2 = nn.Linear(hidden_dim, 1, bias=True)
        self.FC_MG_1 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.FC_MG_2 = nn.Linear(hidden_dim, 1, bias=True)
        self.FC_MN_1 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.FC_MN_2 = nn.Linear(hidden_dim, 1, bias=True)
        self.FC_PO4_1 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.FC_PO4_2 = nn.Linear(hidden_dim, 1, bias=True)
        self.FC_SO4_1 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.FC_SO4_2 = nn.Linear(hidden_dim, 1, bias=True)
        self.FC_ZN_1 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.FC_ZN_2 = nn.Linear(hidden_dim, 1, bias=True)
        self.FC_null1 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.FC_null2 = nn.Linear(hidden_dim, 1, bias=True)

        # Initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def forward(self, protein_feat, mask):
        # Data augmentation
        if self.training and self.augment_eps > 0:
            protein_feat = protein_feat + self.augment_eps * torch.randn_like(protein_feat)

        h_V = self.input_block(protein_feat)
        h_V = self.hidden_block(h_V)

        for layer in self.encoder_layers:
            h_V = layer(h_V, mask)
        
        #print(h_V)
        # dict_keys(['CA', 'CO', 'CU', 'FE2', 'FE', 'MG', 'MN', 'PO4', 'SO4', 'ZN', 'null'])
        logits_CA = self.FC_CA_2(F.leaky_relu(self.FC_CA_1(h_V))).squeeze(-1) # [batch_size, maxlen]
        logits_CO = self.FC_CO_2(F.leaky_relu(self.FC_CO_1(h_V))).squeeze(-1)
        logits_CU = self.FC_CU_2(F.leaky_relu(self.FC_CU_1(h_V))).squeeze(-1)
        logits_FE2 = self.FC_FE2_2(F.leaky_relu(self.FC_FE2_1(h_V))).squeeze(-1)
        logits_FE = self.FC_FE_2(F.leaky_relu(self.FC_FE_1(h_V))).squeeze(-1) # [batch_size, maxlen]
        logits_MG = self.FC_MG_2(F.leaky_relu(self.FC_MG_1(h_V))).squeeze(-1)
        logits_MN = self.FC_MN_2(F.leaky_relu(self.FC_MN_1(h_V))).squeeze(-1)
        logits_PO4 = self.FC_PO4_2(F.leaky_relu(self.FC_PO4_1(h_V))).squeeze(-1)
        logits_SO4 = self.FC_SO4_2(F.leaky_relu(self.FC_SO4_1(h_V))).squeeze(-1) # [batch_size, maxlen]
        logits_ZN = self.FC_ZN_2(F.leaky_relu(self.FC_ZN_1(h_V))).squeeze(-1)
        logits_null = self.FC_null2(F.leaky_relu(self.FC_null1(h_V))).squeeze(-1)

        #print(logits_ZN.shape, logits_CA.shape, logits_MG.shape, logits_MN.shape)
        
        #logits = torch.cat((logits_ZN, logits_CA, logits_MG, logits_MN), 1)
        return logits_CA, logits_CO, logits_CU, logits_FE2, logits_FE, logits_MG, logits_MN, logits_PO4, logits_SO4, logits_ZN, logits_null