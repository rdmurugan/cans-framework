import torch
import torch.nn as nn
from typing import Tuple, Dict, Any

from ..exceptions import DimensionMismatchError, ModelError
from ..validation import ModelValidator, DataValidator

class GatedFusion(nn.Module):
    def __init__(self, gnn_dim: int, bert_dim: int, fused_dim: int, dropout: float = 0.1):
        super().__init__()
        
        if gnn_dim <= 0 or bert_dim <= 0 or fused_dim <= 0:
            raise ValueError("All dimensions must be positive")
        
        self.gnn_dim = gnn_dim
        self.bert_dim = bert_dim
        self.fused_dim = fused_dim
        
        self.gate = nn.Sequential(
            nn.Linear(gnn_dim + bert_dim, fused_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fused_dim, fused_dim),
            nn.Sigmoid()
        )
        self.proj = nn.Linear(gnn_dim + bert_dim, fused_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, gnn_emb: torch.Tensor, bert_emb: torch.Tensor) -> torch.Tensor:
        try:
            # Validate input dimensions
            if gnn_emb.size(-1) != self.gnn_dim:
                raise DimensionMismatchError(f"GNN embedding dim {gnn_emb.size(-1)} != expected {self.gnn_dim}")
            
            if bert_emb.size(-1) != self.bert_dim:
                raise DimensionMismatchError(f"BERT embedding dim {bert_emb.size(-1)} != expected {self.bert_dim}")
            
            if gnn_emb.size(0) != bert_emb.size(0):
                raise DimensionMismatchError(f"Batch size mismatch: GNN {gnn_emb.size(0)} != BERT {bert_emb.size(0)}")
            
            combined = torch.cat([gnn_emb, bert_emb], dim=-1)
            gate = self.gate(combined)
            projected = self.dropout(self.proj(combined))
            return gate * projected
            
        except Exception as e:
            if isinstance(e, DimensionMismatchError):
                raise
            raise ModelError(f"GatedFusion forward failed: {str(e)}")

class CFRNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        
        if input_dim <= 0 or hidden_dim <= 0 or num_layers < 1:
            raise ValueError("input_dim and hidden_dim must be positive, num_layers >= 1")
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Build encoder layers
        layers = []
        layers.append(nn.Linear(input_dim + 1, hidden_dim))  # +1 for treatment
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        self.encoder = nn.Sequential(*layers)
        self.mu0_head = nn.Linear(hidden_dim, 1)
        self.mu1_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            # Validate inputs
            if x.size(-1) != self.input_dim:
                raise DimensionMismatchError(f"Input dim {x.size(-1)} != expected {self.input_dim}")
            
            if x.size(0) != t.size(0):
                raise DimensionMismatchError(f"Batch size mismatch: x {x.size(0)} != t {t.size(0)}")
            
            # Ensure treatment is properly shaped
            if t.dim() == 1:
                t = t.unsqueeze(1)
            elif t.dim() > 2:
                raise DimensionMismatchError(f"Treatment tensor has too many dims: {t.dim()}")
            
            xt = torch.cat([x, t], dim=-1)
            h = self.encoder(xt)
            mu0 = self.mu0_head(h)
            mu1 = self.mu1_head(h)
            
            return mu0, mu1
            
        except Exception as e:
            if isinstance(e, DimensionMismatchError):
                raise
            raise ModelError(f"CFRNet forward failed: {str(e)}")

class CANS(nn.Module):
    def __init__(self, gnn_module, text_encoder, fusion_dim: int = 256, 
                 cfrnet_config: Dict[str, Any] = None, fusion_config: Dict[str, Any] = None):
        super().__init__()
        
        # Validate model compatibility
        ModelValidator.validate_model_compatibility(gnn_module, text_encoder, fusion_dim)
        
        self.gnn = gnn_module
        self.text_encoder = text_encoder
        self.gnn_dim = gnn_module.output_dim
        self.bert_dim = text_encoder.config.hidden_size
        self.fusion_dim = fusion_dim
        
        # Initialize fusion layer with config
        fusion_config = fusion_config or {}
        self.fusion = GatedFusion(
            self.gnn_dim, 
            self.bert_dim, 
            fusion_dim,
            dropout=fusion_config.get('dropout', 0.1)
        )
        
        # Initialize CFRNet with config
        cfrnet_config = cfrnet_config or {}
        self.cfrnet = CFRNet(
            fusion_dim,
            hidden_dim=cfrnet_config.get('hidden_dim', 128),
            num_layers=cfrnet_config.get('num_layers', 2),
            dropout=cfrnet_config.get('dropout', 0.1)
        )

    def forward(self, graph_data, text_input: Dict[str, torch.Tensor], 
                treatment: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            # Validate inputs
            DataValidator.validate_graph_data(graph_data)
            DataValidator.validate_text_data(text_input)
            
            if not isinstance(treatment, torch.Tensor):
                raise ModelError("Treatment must be a torch.Tensor")
            
            # Forward pass through components
            gnn_emb = self.gnn(graph_data)
            
            # Extract CLS token embedding from BERT
            bert_output = self.text_encoder(**text_input)
            bert_emb = bert_output.last_hidden_state[:, 0, :]  # CLS token
            
            # Validate embeddings have correct batch size
            batch_size = treatment.size(0)
            if gnn_emb.size(0) != batch_size:
                raise DimensionMismatchError(f"GNN batch size {gnn_emb.size(0)} != expected {batch_size}")
            if bert_emb.size(0) != batch_size:
                raise DimensionMismatchError(f"BERT batch size {bert_emb.size(0)} != expected {batch_size}")
            
            # Fusion and counterfactual prediction
            fused = self.fusion(gnn_emb, bert_emb)
            mu0, mu1 = self.cfrnet(fused, treatment)
            
            return mu0, mu1
            
        except Exception as e:
            if isinstance(e, (DimensionMismatchError, ModelError)):
                raise
            raise ModelError(f"CANS forward failed: {str(e)}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model configuration and parameter counts"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'gnn_type': self.gnn.__class__.__name__,
            'gnn_dim': self.gnn_dim,
            'bert_dim': self.bert_dim,
            'fusion_dim': self.fusion_dim,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'frozen_text_encoder': not any(p.requires_grad for p in self.text_encoder.parameters())
        }
