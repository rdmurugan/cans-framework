import torch
import torch.nn as nn

class GatedFusion(nn.Module):
    def __init__(self, gnn_dim, bert_dim, fused_dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(gnn_dim + bert_dim, fused_dim),
            nn.ReLU(),
            nn.Linear(fused_dim, fused_dim),
            nn.Sigmoid()
        )
        self.proj = nn.Linear(gnn_dim + bert_dim, fused_dim)

    def forward(self, gnn_emb, bert_emb):
        combined = torch.cat([gnn_emb, bert_emb], dim=-1)
        gate = self.gate(combined)
        projected = self.proj(combined)
        return gate * projected

class CFRNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mu0_head = nn.Linear(hidden_dim, 1)
        self.mu1_head = nn.Linear(hidden_dim, 1)

    def forward(self, x, t):
        xt = torch.cat([x, t.unsqueeze(1)], dim=-1)
        h = self.encoder(xt)
        mu0 = self.mu0_head(h)
        mu1 = self.mu1_head(h)
        return mu0, mu1

class CANS(nn.Module):
    def __init__(self, gnn_module, text_encoder, fusion_dim=256):
        super().__init__()
        self.gnn = gnn_module
        self.text_encoder = text_encoder
        self.gnn_dim = gnn_module.output_dim
        self.bert_dim = text_encoder.config.hidden_size
        self.fusion = GatedFusion(self.gnn_dim, self.bert_dim, fusion_dim)
        self.cfrnet = CFRNet(fusion_dim)

    def forward(self, graph_data, text_input, treatment):
        gnn_emb = self.gnn(graph_data)
        bert_emb = self.text_encoder(**text_input).last_hidden_state[:, 0, :]
        fused = self.fusion(gnn_emb, bert_emb)
        mu0, mu1 = self.cfrnet(fused, treatment)
        return mu0, mu1
