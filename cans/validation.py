"""Validation utilities for the CANS framework"""

import torch
from torch_geometric.data import Data, Batch
from typing import Dict, Any, Tuple, Optional
import numpy as np

from .exceptions import ValidationError, DataError, DimensionMismatchError


class DataValidator:
    """Validates data inputs for CANS models"""
    
    @staticmethod
    def validate_graph_data(graph_data: Data) -> bool:
        """Validate PyTorch Geometric Data object"""
        try:
            # Check required attributes
            if not hasattr(graph_data, 'x'):
                raise DataError("Graph data missing node features 'x'")
            
            if not hasattr(graph_data, 'edge_index'):
                raise DataError("Graph data missing edge indices 'edge_index'")
            
            # Check data types and shapes
            if not isinstance(graph_data.x, torch.Tensor):
                raise DataError("Node features must be torch.Tensor")
            
            if not isinstance(graph_data.edge_index, torch.Tensor):
                raise DataError("Edge indices must be torch.Tensor")
            
            # Check edge_index format
            if graph_data.edge_index.dim() != 2:
                raise DataError(f"Edge index must be 2D, got {graph_data.edge_index.dim()}D")
            
            if graph_data.edge_index.size(0) != 2:
                raise DataError(f"Edge index must have shape [2, num_edges], got {graph_data.edge_index.shape}")
            
            # Check edge indices are within bounds
            num_nodes = graph_data.x.size(0)
            max_edge_idx = graph_data.edge_index.max().item()
            if max_edge_idx >= num_nodes:
                raise DataError(f"Edge index {max_edge_idx} >= num_nodes {num_nodes}")
            
            # Check for negative indices
            if graph_data.edge_index.min().item() < 0:
                raise DataError("Edge indices cannot be negative")
            
            return True
            
        except Exception as e:
            if isinstance(e, DataError):
                raise
            raise DataError(f"Graph data validation failed: {str(e)}")
    
    @staticmethod
    def validate_text_data(text_data: Dict[str, torch.Tensor]) -> bool:
        """Validate tokenized text data"""
        try:
            required_keys = ['input_ids', 'attention_mask']
            
            # Check required keys
            for key in required_keys:
                if key not in text_data:
                    raise DataError(f"Text data missing required key: {key}")
            
            # Check tensor properties
            for key, tensor in text_data.items():
                if not isinstance(tensor, torch.Tensor):
                    raise DataError(f"Text data[{key}] must be torch.Tensor")
                
                if tensor.dim() not in [1, 2]:
                    raise DataError(f"Text tensor {key} must be 1D or 2D, got {tensor.dim()}D")
            
            # Check sequence length consistency
            seq_len = text_data['input_ids'].size(-1)
            for key, tensor in text_data.items():
                if tensor.size(-1) != seq_len:
                    raise DataError(f"Inconsistent sequence length in text data: {key}")
            
            return True
            
        except Exception as e:
            if isinstance(e, DataError):
                raise
            raise DataError(f"Text data validation failed: {str(e)}")
    
    @staticmethod
    def validate_treatment_outcome(treatment: torch.Tensor, outcome: torch.Tensor) -> bool:
        """Validate treatment and outcome tensors"""
        try:
            # Check tensor types
            if not isinstance(treatment, torch.Tensor):
                raise DataError("Treatment must be torch.Tensor")
            
            if not isinstance(outcome, torch.Tensor):
                raise DataError("Outcome must be torch.Tensor")
            
            # Check dimensions
            if treatment.dim() > 2:
                raise DataError(f"Treatment tensor has too many dimensions: {treatment.dim()}")
            
            if outcome.dim() > 2:
                raise DataError(f"Outcome tensor has too many dimensions: {outcome.dim()}")
            
            # Check batch size consistency
            batch_size_t = treatment.size(0)
            batch_size_o = outcome.size(0)
            
            if batch_size_t != batch_size_o:
                raise DataError(f"Treatment and outcome batch sizes don't match: {batch_size_t} vs {batch_size_o}")
            
            # Check treatment values (should be 0 or 1 for binary treatments)
            unique_treatments = torch.unique(treatment)
            valid_treatments = torch.tensor([0.0, 1.0], device=treatment.device)
            
            if not torch.all(torch.isin(unique_treatments, valid_treatments)):
                raise DataError(f"Invalid treatment values. Expected 0 or 1, got: {unique_treatments}")
            
            return True
            
        except Exception as e:
            if isinstance(e, DataError):
                raise
            raise DataError(f"Treatment/outcome validation failed: {str(e)}")
    
    @staticmethod
    def validate_batch(batch: Dict[str, Any]) -> bool:
        """Validate complete data batch"""
        try:
            required_keys = ['graph', 'text', 'treatment', 'outcome']
            
            # Check required keys
            for key in required_keys:
                if key not in batch:
                    raise DataError(f"Batch missing required key: {key}")
            
            # Validate individual components
            DataValidator.validate_graph_data(batch['graph'])
            DataValidator.validate_text_data(batch['text'])
            DataValidator.validate_treatment_outcome(batch['treatment'], batch['outcome'])
            
            return True
            
        except Exception as e:
            if isinstance(e, DataError):
                raise
            raise DataError(f"Batch validation failed: {str(e)}")


class ModelValidator:
    """Validates model configurations and states"""
    
    @staticmethod
    def validate_model_compatibility(gnn_module, text_encoder, fusion_dim: int) -> bool:
        """Validate that model components are compatible"""
        try:
            # Check GNN module
            if not hasattr(gnn_module, 'output_dim'):
                raise ValidationError("GNN module must have 'output_dim' attribute")
            
            if not callable(getattr(gnn_module, 'forward')):
                raise ValidationError("GNN module must have callable 'forward' method")
            
            # Check text encoder
            if not hasattr(text_encoder, 'config'):
                raise ValidationError("Text encoder must have 'config' attribute")
            
            if not hasattr(text_encoder.config, 'hidden_size'):
                raise ValidationError("Text encoder config must have 'hidden_size' attribute")
            
            # Check dimension compatibility
            gnn_dim = gnn_module.output_dim
            bert_dim = text_encoder.config.hidden_size
            
            if gnn_dim <= 0:
                raise ValidationError(f"GNN output dimension must be positive, got {gnn_dim}")
            
            if bert_dim <= 0:
                raise ValidationError(f"Text encoder hidden size must be positive, got {bert_dim}")
            
            if fusion_dim <= 0:
                raise ValidationError(f"Fusion dimension must be positive, got {fusion_dim}")
            
            return True
            
        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            raise ValidationError(f"Model compatibility validation failed: {str(e)}")
    
    @staticmethod
    def validate_model_forward(model, sample_batch: Dict[str, Any]) -> bool:
        """Validate model forward pass with sample data"""
        try:
            model.eval()
            
            with torch.no_grad():
                graph_data = sample_batch['graph']
                text_input = sample_batch['text']
                treatment = sample_batch['treatment']
                
                # Run forward pass
                mu0, mu1 = model(graph_data, text_input, treatment)
                
                # Check outputs
                if not isinstance(mu0, torch.Tensor) or not isinstance(mu1, torch.Tensor):
                    raise ValidationError("Model must return two torch.Tensors (mu0, mu1)")
                
                if mu0.shape != mu1.shape:
                    raise ValidationError(f"mu0 and mu1 must have same shape: {mu0.shape} vs {mu1.shape}")
                
                batch_size = treatment.size(0)
                if mu0.size(0) != batch_size:
                    raise ValidationError(f"Output batch size {mu0.size(0)} doesn't match input {batch_size}")
                
                return True
                
        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            raise ValidationError(f"Model forward validation failed: {str(e)}")


def validate_propensity_overlap(treatment: torch.Tensor, threshold: float = 0.1) -> Tuple[bool, Dict[str, float]]:
    """
    Validate propensity score overlap for causal inference
    
    Returns:
        bool: Whether overlap condition is satisfied
        dict: Overlap statistics
    """
    treatment_np = treatment.cpu().numpy()
    
    prop_score = np.mean(treatment_np)
    
    stats = {
        'propensity_score': prop_score,
        'treatment_ratio': prop_score,
        'control_ratio': 1 - prop_score,
        'min_group_size': min(np.sum(treatment_np), np.sum(1 - treatment_np)),
        'overlap_satisfied': prop_score > threshold and prop_score < (1 - threshold)
    }
    
    return stats['overlap_satisfied'], stats